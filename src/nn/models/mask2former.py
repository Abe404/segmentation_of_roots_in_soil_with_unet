import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

models = ["mask2former"]

class ModelShim(torch.nn.Module):
    """Shim for Mask2Former (or other models) to handle the forward pass and crop the output to 388x388."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()
        # Get the model output
        outputs = self.model(*args, **kwargs)

        # Extract the mask queries logits (segmentation masks)
        masks = outputs.masks_queries_logits  # (batch_size, num_queries, height, width)

        # Crop the output to 388x388
        cropped_masks = self.crop_to_388x388(masks)

        return cropped_masks

    def crop_to_388x388(self, tensor):
        """Crop the tensor to the central 388x388 region."""
        _, _, h, w = tensor.shape
        start_h = (h - 388) // 2
        start_w = (w - 388) // 2
        return tensor[:, :, start_h:start_h + 388, start_w:start_w + 388]


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
    # Reduce stride to preserve resolution
    config.encoder_stride = 1
    # Reduce patch size to keep the resolution high
    config.backbone_config.patch_size = 1
    # Adjust window size to control feature aggregation
    config.backbone_config.window_size = 2

    if pretrained_model:
        # TODO: doesn't work with the custom resolution-preserving config
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            pt_name,
            config=config
        )
    else:
        model = Mask2FormerForUniversalSegmentation(config)
    return ModelShim(model)
