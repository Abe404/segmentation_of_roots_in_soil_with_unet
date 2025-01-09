import torch
import torchvision.models as tv_models
import torchvision.models.segmentation as seg_models


models = [
    "deeplabv3",
    "fcn",
    "lraspp",
]


pretrained_weights = {
    ("deeplabv3", "mobilenet_v3_large"):
        seg_models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
    ("deeplabv3", "resnet101"):
        seg_models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    ("deeplabv3", "resnet50"):
        seg_models.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    ("fcn", "resnet101"):
        seg_models.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    ("fcn", "resnet50"):
        seg_models.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    ("lraspp", "mobilenet_v3_large"):
        seg_models.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
    "mobilenet_v3_large":
        tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    "resnet50":
        tv_models.ResNet50_Weights.IMAGENET1K_V2,
    "resnet101":
        tv_models.ResNet101_Weights.IMAGENET1K_V2
}


class ModelShim(torch.nn.Module):
    """Replace the classifier head with a fresh one with 2 classes, get the
    "out" key in forward, crop to 388x388.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        clfcls = model.classifier.__class__
        if clfcls.__name__ == 'LRASPPHead':
            self.model.classifier = clfcls(40, 960, 2, 128)
        else:
            in_channels = next(model.classifier.parameters()).size(1)
            self.model.classifier = clfcls(in_channels, 2)

    def forward(self, *args, **kwargs):
        out = self.model.forward(*args, **kwargs)["out"]
        return crop_tensor(out, (None, None, 388, 388))


def new(name, encoder_name, pretrained_model, pretrained_backbone):
    if pretrained_backbone:
        weights_backbone = pretrained_weights[encoder_name]
    else:
        weights_backbone = None

    if pretrained_model:
        assert weights_backbone is not None
        weights = pretrained_weights[(name, encoder_name)]
    else:
        weights = None

    model_cls = getattr(seg_models, name + "_" + encoder_name)
    return ModelShim(
        model_cls(weights=weights, weights_backbone=weights_backbone))
