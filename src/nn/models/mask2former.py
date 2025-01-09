import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

from nn.im_utils import crop_to_388x388

models = ["mask2former"]


class ModelShim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.upsample = torch.nn.modules.Upsample(
            scale_factor=4, mode='bilinear')

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        masks = outputs.masks_queries_logits
        cropped_masks = crop_to_388x388(self.upsample(masks))
        return cropped_masks


def new(name, encoder_name, pretrained_model, pretrained_backbone):
    assert (not pretrained_model and not pretrained_backbone) \
        or (pretrained_model and pretrained_backbone)

    assert encoder_name in {
        "swin-tiny-coco-instance",
        "swin-small-coco-instance",
        "swin-base-coco-instance",
        "swin-tiny-ade-semantic",
        "swin-small-ade-semantic",
        "swin-base-ade-semantic",
    }
    pt_name = f"facebook/mask2former-{encoder_name}"

    config = Mask2FormerConfig.from_pretrained(pt_name, num_labels=1)

    if pretrained_model:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pt_name, config=config
        )
    else:
        model = Mask2FormerForUniversalSegmentation(config)
    return ModelShim(model)
