import torch
import torch.nn.functional as F
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
        # import ipdb; ipdb.set_trace()
        # Get the model output
        outputs = self.model(*args, **kwargs)

        # Extract the mask queries logits (segmentation masks)
        masks = outputs.masks_queries_logits  # (batch_size, num_queries, height, width)

        input_shape = args[0].shape[-2:]  # Get height and width from input shape

        # Resize masks to match input shape
        masks_resized = F.interpolate(masks, size=input_shape, mode="bilinear", align_corners=False)
                
        # Crop the output to 388x388
        cropped_masks = self.crop_to_388x388(masks_resized)

        return cropped_masks

    def crop_to_388x388(self, tensor):
        """Crop the tensor to the central 388x388 region."""
        _, _, h, w = tensor.shape
        start_h = (h - 388) // 2
        start_w = (w - 388) // 2
        cropped = tensor[:, :, start_h:start_h + 388, start_w:start_w + 388]
        assert cropped.shape[-2:] == (388, 388), f"shape should be 388,388, shape is {cropped.shape}"
        return cropped


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
