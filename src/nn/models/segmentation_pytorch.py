import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from nn.im_utils import crop_tensor


model_mapping = {
    "unet++": smp.UnetPlusPlus,
    "deeplabv3+": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "linknet": smp.Linknet,         # Add Linknet
    "pspnet": smp.PSPNet            # Add PSPNet
}


models = list(model_mapping.keys())


class ModelShim(torch.nn.Module):
    """This shim pads the input image to the nearest multiple of 32 (if needed)
    for segmentation_models_pytorch models and crops the output back to 388x388.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Pad input to be divisible by 32
        height, width = x.shape[2], x.shape[3]
        pad_height = (32 - height % 32) % 32
        pad_width = (32 - width % 32) % 32

        x_padded = F.pad(x, (0, pad_width, 0, pad_height))

        # Forward pass through the model
        out = self.model(x_padded)

        # Crop the output back to 388x388
        return crop_tensor(out, (None, None, 388, 388))


def new(name, encoder_name, pretrained_model, pretrained_backbone):
    model_cls = model_mapping[name]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained_backbone else None,
        in_channels=3,
        classes=1,
        activation=None
    )
