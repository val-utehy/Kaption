import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import yaml

from models.kaption import kaption_decoder
from data.flickr8k_dataset import flickr8k_caption_eval
from data.utils import flickr8k_caption_eval as compute_metrics


@torch.no_grad()
def evaluate(model, loader, device, config):
    model.eval()
    result = []
    for i, (image, img_ids) in enumerate(loader):
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
        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(loader)}]')
    return result


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading KAPTION model...')
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
    )
    model = model.to(device)

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = flickr8k_caption_eval(
        transform_test, config['image_root'], config['ann_root'], args.split
    )
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        num_workers=4, pin_memory=True, shuffle=False)

    print(f'Evaluating KAPTION on Flickr8k-{args.split} ({len(dataset)} images)...')
    result = evaluate(model, loader, device, config)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(args.output_dir, f'kaption_{args.split}_result.json')
    json.dump(result, open(result_file, 'w'))
    print(f'\nResults saved to {result_file}')

    print('\n--- Metrics ---')
    compute_metrics(config['ann_root'], result_file, args.split)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/home/kienngyuen/PycharmProjects/Multimodel-research/kaption-version-1/output/KAPTION_flickr8k/checkpoint_best.pth', help='path to finetuned kaption checkpoint')
    parser.add_argument('--config', default='/home/kiennguyen/source/multimodel/kaption-version-1/configs/kaption_flickr8k.yaml')
    parser.add_argument('--split', default='test', choices=['val', 'test'])
    parser.add_argument('--output_dir', default='./output/kaption_flickr8k')
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.checkpoint:
        config['pretrained'] = args.checkpoint

    main(args, config)