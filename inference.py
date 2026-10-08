import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.kaption import kaption_decoder

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def build_transform(image_size):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])


def collect_image_paths(image_arg):
    path = Path(image_arg)
    if path.is_file():
        return [path]
    if path.is_dir():
        paths = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise ValueError(f'No images found in directory: {path}')
        return paths
    raise ValueError(f'--image path does not exist: {path}')


def load_batch(paths, transform):
    images, kept_paths = [], []
    for p in paths:
        try:
            image = Image.open(p).convert('RGB')
        except Exception as e:
            print(f'  [skip] failed to open {p}: {e}')
            continue
        images.append(transform(image))
        kept_paths.append(p)
    if not images:
        return None, []
    return torch.stack(images, dim=0), kept_paths


@torch.no_grad()
def run_inference(model, image_paths, transform, device, batch_size, gen_kwargs):
    results = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images, kept_paths = load_batch(batch_paths, transform)
        if images is None:
            continue
        images = images.to(device)
        captions = model.generate(images, **gen_kwargs)
        for p, caption in zip(kept_paths, captions):
            print(f'{p.name} -> {caption}')
            results.append({'file_name': p.name, 'caption': caption})
    return results


def main(args, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    image_paths = collect_image_paths(args.image)
    print(f'Found {len(image_paths)} image(s) to caption')

    print('Loading KAPTION model...')
    model = kaption_decoder(
        pretrained=args.checkpoint,
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
    model.eval()

    transform = build_transform(config['image_size'])

    batch_size = args.batch_size or config['batch_size']
    gen_kwargs = dict(
        sample=args.sample,
        num_beams=args.num_beams or config['num_beams'],
        max_length=args.max_length or config['max_length'],
        min_length=args.min_length or config['min_length'],
        repetition_penalty=1.1,
    )

    results = run_inference(model, image_paths, transform, device, batch_size, gen_kwargs)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_file = os.path.join(args.output_dir, 'inference_result.json')
    json.dump(results, open(result_file, 'w'), ensure_ascii=False, indent=2)
    print(f'\nResults saved to {result_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/kaption_nocaps.yaml')
    parser.add_argument('--checkpoint', required=True, help='path to finetuned KAPTION checkpoint')
    parser.add_argument('--image', required=True, help='path to a single image file or a directory of images')
    parser.add_argument('--output_dir', default='output/results')
    parser.add_argument('--batch_size', type=int, default=None, help='override config batch_size')
    parser.add_argument('--num_beams', type=int, default=None, help='override config num_beams')
    parser.add_argument('--max_length', type=int, default=None, help='override config max_length')
    parser.add_argument('--min_length', type=int, default=None, help='override config min_length')
    parser.add_argument('--sample', action='store_true', help='use sampling instead of beam search')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    main(args, config)
