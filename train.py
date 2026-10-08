import argparse
import os
import json
import random
import time
import datetime
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import utils
from models.kaption import kaption_decoder
from data.flickr8k_dataset import flickr8k_train, flickr8k_caption_eval
from data.coco_dataset import coco_train, coco_caption_eval
from data.coco_karpathy_dataset2 import karpathy_train, karpathy_caption_eval
from data.utils import flickr8k_caption_eval as flickr8k_metrics
from data.utils import coco2017_caption_eval as coco_metrics
from data.utils import coco_caption_eval as karpathy_metrics
from transform.randaugment import RandomAugment


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cosine_lr_schedule(optimizer, epoch, max_epoch, config):
    """Cosine decay with 1-epoch linear warmup, per param-group."""
    warmup_epochs = config.get('warmup_epochs', 1)
    for pg in optimizer.param_groups:
        init_lr = float(pg['init_lr'])
        min_lr  = float(pg.get('min_lr', 0.0))
        if epoch < warmup_epochs:
            lr = init_lr * (epoch + 1) / warmup_epochs
        else:
            t = epoch - warmup_epochs
            T = max_epoch - warmup_epochs
            lr = min_lr + (init_lr - min_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * t / T)).item())
        pg['lr'] = lr


def save_result_local(result, result_dir, filename, remove_duplicate='image_id'):
    final_path = os.path.join(result_dir, f'{filename}.json')
    if remove_duplicate:
        seen, deduped = set(), []
        for r in result:
            key = r[remove_duplicate]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        result = deduped
    json.dump(result, open(final_path, 'w'))
    return final_path


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, epoch, device):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr_resampler', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('lr_decoder', utils.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    for image, caption, _ in metric_logger.log_every(loader, 50, f'Epoch [{epoch}]'):
        image = image.to(device)
        loss = model(image, caption)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr_resampler=optimizer.param_groups[0]['lr'])
        metric_logger.update(lr_decoder=optimizer.param_groups[1]['lr'])

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger.global_avg())
    return {k: f'{m.global_avg:.4f}' for k, m in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    result = []
    for image, img_ids in loader:
        image = image.to(device)
        captions = model.generate(
            image,
            sample=False,
            num_beams=config['num_beams'],
            max_length=config['max_length'],
            min_length=config['min_length'],
        )
        for caption, img_id in zip(captions, img_ids.tolist()):
            result.append({'image_id': img_id, 'caption': caption})
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # ---- transforms -------------------------------------------------------
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            config['image_size'], scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 5, isPIL=True, augs=[
            'Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate',
        ]),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(
            (config['image_size'], config['image_size']),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    # ---- datasets ---------------------------------------------------------
    dataset_name = config.get('dataset', 'flickr8k')

    print('Creating Flickr8k datasets...')
    train_dataset = flickr8k_train(
        transform_train, config['image_root'], config['ann_root'],
        prompt=config['prompt'],
    )
    val_dataset  = flickr8k_caption_eval(transform_test, config['image_root'], config['ann_root'], 'val')
    test_dataset = flickr8k_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')

    compute_metrics = lambda result_file, split: flickr8k_metrics(config['ann_root'], result_file, split)

    

    if args.distributed:
        num_tasks   = utils.get_world_size()
        global_rank = utils.get_rank()
        from data import create_sampler, create_loader
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset],
            [True, False, False], num_tasks, global_rank,
        )
    else:
        samplers = [None, None, None]

    bs = config['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=bs, sampler=samplers[0],
        shuffle=(samplers[0] is None), num_workers=4,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, sampler=samplers[1],
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=bs, sampler=samplers[2],
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f'  train={len(train_dataset)}  val={len(val_dataset)}  test={len(test_dataset)}')

    # ---- model ------------------------------------------------------------
    print('Creating KAPTION model...')
    model = kaption_decoder(
        pretrained=config['pretrained'],
        image_size=config['image_size'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer'],
        prompt=config['prompt'],
        num_latent_tokens=config['num_latent_tokens'],
        mot_num_layers=config['mot_num_layers'],
        use_rope=config['use_rope'],
        gamma_init=float(config.get('gamma_init', -6.0)),
    )
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    resampler_params, anchor_params, gamma_params = [], [], []
    for n, p in model_without_ddp.latent_resampler.named_parameters():
        if n.endswith('anchors'):
            anchor_params.append(p)
        elif n.endswith('log_gamma'):
            gamma_params.append(p)
        else:
            resampler_params.append(p)
    decoder_params = list(model_without_ddp.caption_decoder.parameters())
    # image_encoder is frozen (freeze_vit_epochs=0) — excluded from optimizer

    init_lr = float(config['init_lr'])
    min_lr  = float(config['min_lr'])
    anchor_lr = float(config.get('anchor_lr', init_lr * 10))
    gamma_lr  = float(config.get('gamma_lr',  init_lr * 10))
    optimizer = torch.optim.AdamW(
        [
            {'params': resampler_params, 'lr': init_lr * 10,
             'init_lr': init_lr * 10,  'min_lr': min_lr * 10},
            {'params': decoder_params, 'lr': init_lr,
             'init_lr': init_lr,        'min_lr': min_lr},
            {'params': anchor_params,  'lr': anchor_lr,
             'init_lr': anchor_lr,      'min_lr': 0.0, 'weight_decay': 0.0},
            {'params': gamma_params,   'lr': gamma_lr,
             'init_lr': gamma_lr,       'min_lr': 0.0, 'weight_decay': 0.0},
        ],
        weight_decay=config['weight_decay'],
    )
    print(f"optimizer groups: resampler={len(resampler_params)} decoder={len(decoder_params)} "
          f"anchors={len(anchor_params)} (lr {anchor_lr:g}, wd 0) "
          f"log_gamma={len(gamma_params)} (lr {gamma_lr:g}, wd 0)")

    # ---- training loop ----------------------------------------------------
    best_cider, best_epoch, no_improve = 0.0, 0, 0
    patience = int(config.get('early_stopping_patience', 4))
    print('Start training')
    start = time.time()
    if utils.is_main_process():
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(f'\n{"="*40} NEW RUN {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {"="*40}\n')
        log_path = os.path.join(args.output_dir, 'train_log.txt')
        with open(log_path, 'w') as f:
            f.write(f"{'Epoch':>5s} | {'Loss':>7s} | "
                    f"{'CIDEr':>7s} | {'Bleu_4':>7s} | "
                    f"{'METEOR':>7s} | {'ROUGE_L':>7s}\n"
                    + '-' * 65 + '\n')

    freeze_vit_epochs = int(config.get('freeze_vit_epochs', -1))

    for epoch in range(config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            if freeze_vit_epochs == -1:
                vit_frozen = False
            elif freeze_vit_epochs == 0:
                vit_frozen = True
            else:
                vit_frozen = epoch < freeze_vit_epochs

            for p in model_without_ddp.image_encoder.parameters():
                p.requires_grad = not vit_frozen

            if vit_frozen != getattr(model_without_ddp, '_vit_frozen_prev', None):
                state = 'frozen' if vit_frozen else 'unfrozen'
                print(f'Epoch {epoch}: image_encoder {state}')
                model_without_ddp._vit_frozen_prev = vit_frozen

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config)
            train_stats = train(model, train_loader, optimizer, epoch, device)

        # eval
        val_result  = evaluate(model_without_ddp, val_loader,  device, config)
        test_result = evaluate(model_without_ddp, test_loader, device, config)

        if utils.is_main_process():
            val_file  = save_result_local(val_result,  args.result_dir, f'val_epoch{epoch}')
            test_file = save_result_local(test_result, args.result_dir, f'test_epoch{epoch}')

            print(f'\n=== Epoch {epoch} — val ===')
            val_eval  = compute_metrics(val_file,  'val')
            print(f'=== Epoch {epoch} — test ===')
            test_eval = compute_metrics(test_file, 'test')

            if args.evaluate:
                log = {
                    **{f'val_{k}':  v for k, v in val_eval.eval.items()},
                    **{f'test_{k}': v for k, v in test_eval.eval.items()},
                }
                with open(os.path.join(args.output_dir, 'evaluate.txt'), 'a') as f:
                    f.write(json.dumps(log) + '\n')
            else:
                val_cider  = val_eval.eval['CIDEr']
                val_bleu4  = val_eval.eval['Bleu_4']
                val_meteor = val_eval.eval['METEOR']
                val_rouge  = val_eval.eval['ROUGE_L']

                if val_cider > best_cider:
                    best_cider, best_epoch, no_improve = val_cider, epoch, 0
                    torch.save(
                        {
                            'model':     model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config':    config,
                            'epoch':     epoch,
                        },
                        os.path.join(args.output_dir, 'checkpoint_best.pth'),
                    )
                    print(f'  *** New best at epoch {epoch} —'
                          f' CIDEr={val_cider:.4f}  Bleu_4={val_bleu4:.4f}'
                          f'  METEOR={val_meteor:.4f}  ROUGE_L={val_rouge:.4f} ***')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f'\nEarly stopping at epoch {epoch}.')
                        print(f'Suggestion: set max_epoch = {best_epoch + 1} for future runs.')
                        break

                # spatial-prior diagnostics: gamma must actually move, and the
                # anchors must not just shrink toward the origin
                with torch.no_grad():
                    gammas = [F.softplus(m.log_gamma).item()
                              for m in model_without_ddp.latent_resampler.modules()
                              if hasattr(m, 'log_gamma')]
                    anc = model_without_ddp.latent_resampler.anchors.detach().float().cpu()
                    anc_d = torch.cdist(anc, anc)
                    anc_d.fill_diagonal_(float('inf'))

                log = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'val_{k}':   v for k, v in val_eval.eval.items()},
                    **{f'test_{k}':  v for k, v in test_eval.eval.items()},
                    'gamma': [round(g, 5) for g in gammas],
                    'anchor_centroid': [round(v, 4) for v in anc.mean(0).tolist()],
                    'anchor_nn_mean': round(anc_d.min(1).values.mean().item(), 4),
                    'epoch': epoch, 'best_epoch': best_epoch,
                }
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log) + '\n')

                train_loss = train_stats.get('loss', 'N/A')
                best_mark  = ' <-- best' if epoch == best_epoch else ''
                row = (
                    f"Epoch {epoch:3d} | "
                    f"Loss: {train_loss:>7s} | "
                    f"CIDEr: {val_cider:.4f} | "
                    f"Bleu_4: {val_bleu4:.4f} | "
                    f"METEOR: {val_meteor:.4f} | "
                    f"ROUGE_L: {val_rouge:.4f} | "
                    f"gamma: {'/'.join(f'{g:.3f}' for g in gammas)} | "
                    f"anc_nn: {anc_d.min(1).values.mean().item():.4f}"
                    f"{best_mark}\n"
                )
                with open(os.path.join(args.output_dir, 'train_log.txt'), 'a') as f:
                    f.write(row)

        if args.evaluate:
            break
        if args.distributed:
            dist.barrier()

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start)))
    print(f'Training time: {elapsed}  |  Best epoch: {best_epoch}')
    print(f'Suggestion: set max_epoch = {best_epoch + patience + 1} for future runs'
          f' (best_epoch={best_epoch}, patience={patience})')
    if utils.is_main_process():
        log_path = os.path.join(args.output_dir, 'train_log.txt')
        with open(log_path, 'a') as f:
            f.write('-' * 65 + '\n')
            f.write(f'Best epoch: {best_epoch}  |  Best CIDEr: {best_cider:.4f}  |  '
                    f'Training time: {elapsed}\n')
            f.write(f'Suggestion: set max_epoch = {best_epoch + patience + 1}\n')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='./configs/kaption_flickr8k.yaml')
    parser.add_argument('--output_dir', default='output/KAPTION_flickr8k')
    parser.add_argument('--evaluate',   action='store_true')
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--seed',       default=42, type=int)
    parser.add_argument('--world_size', default=1,  type=int)
    parser.add_argument('--dist_url',   default='env://')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)