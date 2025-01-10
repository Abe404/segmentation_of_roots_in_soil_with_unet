import torch
import torch.nn.functional as F
from torch.nn.modules.sparse import Embedding
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

from nn.im_utils import crop_to_388x388

models = ["mask2former"]


class ModelShim(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).masks_queries_logits
        input_shape = args[0].shape[-2:]
        logits = F.interpolate(logits, size=input_shape, mode="bilinear")
        return crop_to_388x388(logits)


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

    config = Mask2FormerConfig.from_pretrained(
        pt_name, num_queries=1, num_labels=1)

    if pretrained_model:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(pt_name)
        model.model.transformer_module.queries_embedder = Embedding(
            1, model.model.transformer_module.queries_embedder.embedding_dim)
        model.model.transformer_module.queries_features = Embedding(
            1, model.model.transformer_module.queries_features.embedding_dim)
    else:
        model = Mask2FormerForUniversalSegmentation(config)
    return ModelShim(model)
