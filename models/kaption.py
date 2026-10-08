import os

import torch
import torch.nn as nn
from timm.models.hub import download_cached_file

from models.blip import create_vit, init_tokenizer, is_url
from models.latent_resampler import LatentResampler
from models.med import BertConfig, BertLMHeadModel
from models.vit import interpolate_pos_embed

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MED_CONFIG = os.path.join(_REPO_ROOT, 'configs', 'med_config.json')

# Key prefixes of checkpoints saved before the modules were renamed: BLIP's released
# weights use the first two, earlier KAPTION runs all of them. Most specific first.
_LEGACY_KEY_PREFIXES = (
    ('visual_encoder.', 'image_encoder.'),
    ('text_decoder.', 'caption_decoder.'),
    ('mot_bridge.latent_tokens', 'latent_resampler.latent_query'),
    ('mot_bridge.', 'latent_resampler.'),
)


def _rename_legacy_key(key):
    for old, new in _LEGACY_KEY_PREFIXES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


class KAPTION_Decoder(nn.Module):
    """
    KAPTION-v1: BLIP captioning decoder enhanced with
      - Latent resampler (LatentResampler): compresses ViT tokens → N latent tokens
      - RoPE: replaces absolute position embeddings in the caption decoder

    image_encoder → latent_resampler → caption_decoder, as in the paper's architecture figure.
    """

    def __init__(
        self,
        med_config=_DEFAULT_MED_CONFIG,
        image_size=384,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        prompt='a picture of ',
        num_latent_tokens=32,
        mot_num_layers=2,
        use_rope=True,
        gamma_init=-6.0,
    ):
        super().__init__()

        self.image_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()

        cfg = BertConfig.from_json_file(med_config)
        cfg.add_cross_attention = True
        cfg.encoder_width = cfg.hidden_size   # latent tokens have same dim as text hidden
        cfg.use_rope = use_rope

        self.latent_resampler = LatentResampler(
            num_latent_tokens=num_latent_tokens,
            d_model=cfg.hidden_size,
            vision_width=vision_width,
            num_heads=cfg.num_attention_heads,
            num_layers=mot_num_layers,
            num_patches_side=image_size // 16,
            gamma_init=gamma_init,
        )

        self.caption_decoder = BertLMHeadModel(config=cfg)

        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # map legacy checkpoint keys (see _LEGACY_KEY_PREFIXES) onto the current names
        for key in [k for k in state_dict if k.startswith(prefix)]:
            new_key = prefix + _rename_legacy_key(key[len(prefix):])
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # ------------------------------------------------------------------
    def forward(self, image, caption):
        image_embeds = self.image_encoder(image)                # t_e: [B, N_vis, vision_width]

        latent_embeds = self.latent_resampler(image_embeds)     # q_z: [B, N_lat, hidden]
        latent_atts = torch.ones(
            latent_embeds.shape[:2], dtype=torch.long, device=image.device
        )

        text = self.tokenizer(
            caption, padding='longest', truncation=True,
            max_length=40, return_tensors='pt',
        ).to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        targets = text.input_ids.masked_fill(
            text.input_ids == self.tokenizer.pad_token_id, -100
        )
        targets[:, :self.prompt_length] = -100

        out = self.caption_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=latent_embeds,
            encoder_attention_mask=latent_atts,
            labels=targets,
            return_dict=True,
        )
        return out.loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, image, sample=False, num_beams=3,
                 max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.image_encoder(image)
        latent_embeds = self.latent_resampler(image_embeds)

        latent_atts = torch.ones(
            latent_embeds.shape[:2], dtype=torch.long, device=image.device
        )
        model_kwargs = {
            'encoder_hidden_states': latent_embeds,
            'encoder_attention_mask': latent_atts,
        }

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            outputs = self.caption_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs,
            )
        else:
            outputs = self.caption_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs,
            )

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions


# ----------------------------------------------------------------------

def load_checkpoint(model, url_or_filename):
    """models.blip.load_checkpoint for KAPTION's module names; accepts legacy keys too."""
    if is_url(url_or_filename):
        path = download_cached_file(url_or_filename, check_hash=False, progress=True)
    elif os.path.isfile(url_or_filename):
        path = url_or_filename
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = {_rename_legacy_key(k): v for k, v in checkpoint['model'].items()}

    state_dict['image_encoder.pos_embed'] = interpolate_pos_embed(
        state_dict['image_encoder.pos_embed'], model.image_encoder)
    for key, value in model.state_dict().items():
        if key in state_dict and state_dict[key].shape != value.shape:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f'load checkpoint from {url_or_filename}')
    return model, msg


def kaption_decoder(pretrained='', **kwargs):
    model = KAPTION_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        # latent_resampler and rotary_emb are new — expected to be missing from BLIP checkpoint
        unexpected = [k for k in msg.missing_keys
                      if 'latent_resampler' not in k and 'rotary_emb' not in k]
        if unexpected:
            print('Unexpected missing keys:', unexpected)
    return model