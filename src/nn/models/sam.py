import torch
import torch.nn.functional as F
import transformers

from nn.im_utils import crop_to_388x388

models = ["sam", "sam2", "sam2.1"]


class ModelShim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.image_size = self.model.vision_encoder.image_size

    def forward(self, *args, **kwargs):
        outputs = self.model(*(map(self.preprocess, args)), **kwargs)
        masks = outputs.pred_masks.squeeze(1)
        return self.postprocess(masks)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, (self.image_size, self.image_size), mode="bilinear"
        )
        return x

    def postprocess(self, masks):
        masks = F.interpolate(
            masks, (self.image_size, self.image_size), mode="bilinear",
        )
        return crop_to_388x388(masks)


def new(name, encoder_name, pretrained_model, pretrained_backbone):
    assert (not pretrained_model and not pretrained_backbone) \
        or (pretrained_model and pretrained_backbone)

    if name == "sam":
        assert encoder_name in {"vit-base", "vit-huge"}
    else:
        assert encoder_name in {
            "hiera-tiny",
            "hiera-small",
            "hiera-base_plus",
            "hiera-large",
        }
    pt_name = f"facebook/{name}-{encoder_name}"
    config = transformers.AutoConfig.from_pretrained(pt_name, num_labels=1)
    if pretrained_model:
        if name == "sam":
            model = transformers.SamModel.from_pretrained(pt_name)
        else:
            # TODO
            import ipdb; ipdb.set_trace()
    else:
        if name == "sam":
            model = transformers.SamModel(config)
        else:
            # TODO
            import ipdb; ipdb.set_trace()
    return ModelShim(model)
