import re

import torch
import torch.nn.functional as F
import transformers
import segment_anything as sam
from transformers.models.sam.convert_sam_to_hf import KEYS_TO_MODIFY_MAPPING

from nn.im_utils import crop_to_388x388
from nn.sys_utils import get_device

models = ["sam"]


def build_sam(pt_name):
    if pt_name.endswith("vit-base"):
        fb_model = sam.build_sam_vit_b()
    elif pt_name.endswith("vit-large"):
        fb_model = sam.build_sam_vit_l()
    elif pt_name.endswith("vit-huge"):
        fb_model = sam.build_sam_vit_h()
    else:
        raise ValueError(f"Unsupported model {pt_name}")
    return fb_model


def build_sam_hf(pt_name):
    """Copy weights from hf to original sam model. Basically the reverse of
    transformers/models/sam/convert_sam_to_hf.py."""
    config = transformers.AutoConfig.from_pretrained(pt_name)
    hf_model = transformers.SamModel.from_pretrained(pt_name, config=config)
    fb_model = build_sam(pt_name)

    hf_state = hf_model.state_dict()
    fb_state = fb_model.state_dict()

    fb_state["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"] = \
        hf_state.pop("shared_image_embedding.positional_embedding")

    pattern1 = \
            r".*.output_hypernetworks_mlps.(\d+).(?:(proj_[inout]+)|layers.(\d+)).*"  # noqa: E127, E501
    pattern2 = \
        r".*.(?:output_hypernetworks_mlps.\d+|iou_prediction_head).blocks.*"

    keys_to_mod = {v: k for k, v in KEYS_TO_MODIFY_MAPPING.items()}
    keys_to_mod["not_a_point_embeddings"] = "not_a_point_embed"
    keys_to_mod["transformer.blocks"] = "transformer.layers"
    for key, value in hf_state.items():
        for key_to_modify, new_key in keys_to_mod.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        if (match := re.match(pattern1, key)):
            layer = match.group(2)
            if layer == "proj_in":
                key = key.replace("proj_in", "layers.0")
            elif layer == "0":
                key = key.replace("layers.0", "layers.1")
            elif layer == "proj_out":
                key = key.replace("proj_out", "layers.2")

        if re.match(pattern2, key):
            key = key.replace("blocks", "layers")

        fb_state[key] = value

    fb_model.load_state_dict(fb_state)
    return fb_model


class ModelShim(torch.nn.Module):
    """As close as possible to the original implementation
    (facebook/segment_anything)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        dev = get_device()
        self.image_size = self.model.image_encoder.img_size
        # From facebook/segment_anything
        self.pixel_mean = \
            torch.tensor([123.675, 116.28, 103.53]).to(dev).view(1, -1, 1, 1)
        self.pixel_std = \
            torch.tensor([58.395, 57.12, 57.375]).to(dev).view(1, -1, 1, 1)

    def forward(self, x):
        x_scaled = self.preprocess(x)
        features = self.model.image_encoder(x_scaled)
        prompt_embed = self.model.prompt_encoder(None, None, None)
        dense_pe = self.model.prompt_encoder.get_dense_pe()
        masks, _ = \
            self.model.mask_decoder(features, dense_pe, *prompt_embed, False)
        return self.postprocess(masks)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes x is in (-0.5, 0.5)
        x = torch.clip((x+0.5) * 256, min=0, max=255).type(torch.uint8)
        x = (x - self.pixel_mean) / self.pixel_std
        return F.interpolate(
            x, (self.image_size, self.image_size), mode="bilinear"
        )

    def postprocess(self, masks):
        masks = F.interpolate(
            masks, (self.image_size, self.image_size), mode="bilinear",
        )
        return crop_to_388x388(masks)


def new(name, encoder_name, pretrained_model, pretrained_backbone):
    assert (not pretrained_model and not pretrained_backbone) \
        or (pretrained_model and encoder_name)

    assert encoder_name in {"vit-base", "vit-huge"}
    pt_name = f"facebook/{name}-{encoder_name}"
    if pretrained_model:
        model = build_sam_hf(pt_name)
    else:
        model = build_sam(pt_name)
    return ModelShim(model)
