# KAPTION

## Overview

KAPTION (version 1) is an image captioning model built on the BLIP captioning decoder. It inserts a latent resampler between the image encoder and the caption decoder and replaces the decoder's absolute position embeddings with rotary position embeddings (RoPE):

```text
image_encoder (ViT-B) → latent_resampler (32 latent tokens) → caption_decoder (BERT + RoPE)
```

The latent resampler compresses the ViT patch tokens into a fixed set of latent tokens using spatially biased cross-attention, where each latent has a learned 2D anchor that pulls it toward a distinct image region.

- **Code:** [GitHub repository](https://github.com/val-utehy/Kaption)
- **Weights:** [Hugging Face model](https://huggingface.co/KienNgyuen/kaption-v1)

The model is defined in [`kaption.py`](models/kaption.py), with the resampler in [`latent_resampler.py`](models/latent_resampler.py) and RoPE in [`rope.py`](models/rope.py). The image encoder and caption decoder are initialized from BLIP's `model_base_caption_capfilt_large` checkpoint, which is downloaded automatically on the first training run.

## Setup

Clone the repository and create a Python 3.12 environment:

```bash
git clone https://github.com/val-utehy/Kaption.git
cd Kaption
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install "timm==1.0.28" "transformers==5.14.1" "fairscale==0.4.13" \
  pycocotools pycocoevalcap pyyaml huggingface_hub
```

On Windows, activate the environment with `.venv\Scripts\activate`. If the default PyTorch wheel does not match your CUDA driver, install the matching build from [pytorch.org](https://pytorch.org/get-started/locally/).

These versions were tested with `torch==2.12.1` and `torchvision==0.27.1`. The pins in [`requirements.txt`](requirements.txt) are older and do not match the current code (`transformers==4.15.0` lacks `transformers.generation`), so install with the commands above.

The METEOR metric needs Java on your `PATH`. Without it, evaluation still runs and reports BLEU, ROUGE-L, and CIDEr.

## Dataset

The default configuration trains on Flickr8k. Point `image_root` and `ann_root` in [`kaption_flickr8k.yaml`](configs/kaption_flickr8k.yaml) to your local copy:

```text
flickr8k_dataset/            # image_root
├── 1000268201_693b08cb0e.jpg
└── ...
flickr8k_text/               # ann_root
├── Flickr8k.token.txt
├── Flickr_8k.trainImages.txt
├── Flickr_8k.devImages.txt
└── Flickr_8k.testImages.txt
```

```yaml
image_root: '/absolute/path/to/flickr8k_dataset'
ann_root: '/absolute/path/to/flickr8k_text'
```

The COCO-format ground-truth files `flickr8k_val_gt.json` and `flickr8k_test_gt.json` are generated in `ann_root` on the first run.

## Training

Train KAPTION on Flickr8k:

```bash
python train.py \
  --config configs/kaption_flickr8k.yaml \
  --output_dir output/KAPTION_flickr8k
```

The configuration trains for up to 100 epochs with 10 warmup epochs, batch size 32, and early stopping when validation CIDEr does not improve for 10 epochs. Lower `batch_size` in the config if you run out of GPU memory. Training uses CUDA by default; pass `--device cpu` for CPU execution.

The best checkpoint by validation CIDEr and per-epoch caption results are saved to:

```text
output/KAPTION_flickr8k/checkpoint_best.pth
output/KAPTION_flickr8k/result/val_epoch*.json
output/KAPTION_flickr8k/result/test_epoch*.json
```

## Prediction

Caption a single image or every image in a directory with your trained checkpoint:

```bash
python inference.py \
  --config configs/kaption_flickr8k.yaml \
  --checkpoint output/KAPTION_flickr8k/checkpoint_best.pth \
  --image /absolute/path/to/images
```

Each caption is printed as it is generated, and all results are saved to `output/results/inference_result.json`. Use `--output_dir` to change the location, `--num_beams`, `--max_length`, or `--min_length` to override the generation settings in the config, and `--sample` to use sampling instead of beam search.

To evaluate the checkpoint on the Flickr8k test split (the script name contains a space, so keep the quotes):

```bash
python "eval_flickr8k .py" \
  --config configs/kaption_flickr8k.yaml \
  --checkpoint output/KAPTION_flickr8k/checkpoint_best.pth \
  --split test --output_dir output/eval_flickr8k
```

## Trained models

A checkpoint fine-tuned on Flickr8k (about 2.2 GB) is available on Hugging Face. Download it and run inference without training:

```bash
hf download KienNgyuen/kaption-v1 checkpoint_best.pth --local-dir weights
python inference.py \
  --config configs/kaption_flickr8k.yaml \
  --checkpoint weights/checkpoint_best.pth \
  --image /absolute/path/to/images
```

| Checkpoint | Purpose |
| --- | --- |
| `checkpoint_best.pth` | Best checkpoint by validation CIDEr; use for evaluation and prediction. |


## Acknowledgement

The codebase is built upon BLIP. We sincerely thank their contribution 
