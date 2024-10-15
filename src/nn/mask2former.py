import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

class Mask2FormerShim(torch.nn.Module):
    """Shim for Mask2Former (or other models) to handle the forward pass and crop the output to 388x388."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
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



def get_mask2former_model(pretrained=True):
    """
    Returns a Mask2Former model wrapped in Mask2FormerShim for binary segmentation.
    """
   
    # Load configuration for binary segmentation (single label for foreground/background)
    config = Mask2FormerConfig.from_pretrained(
        "facebook/mask2former-swin-small-ade-semantic",
        num_labels=1  # Binary segmentation only requires one label (foreground/background)
    )

    # Adjust the number of queries to 1 (since we only need one mask)
    config.num_queries = 2  # One query for binary segmentation

    # Modify the backbone and encoder to preserve spatial resolution
    config.encoder_stride = 1  # Reduce stride to preserve resolution
    config.backbone_config.patch_size = 1  # Reduce patch size to keep the resolution high
    config.backbone_config.window_size = 2  # Adjust window size to control feature aggregation

    # Load the model with the custom configuration
    model = Mask2FormerForUniversalSegmentation(config)

    # Wrap the model in the shim for binary segmentation
    model = Mask2FormerShim(model)
    
    return model

# Usage Example
if __name__ == "__main__":
    # Load and preprocess the image
    image_path = "your_image_path.jpg"
    inputs = preprocess_image(image_path)

    # Get the binary segmentation model
    model = get_binary_segmentation_model()

    # Forward pass
    with torch.no_grad():
        predictions = model(**inputs)
    
    # Check the output shape
    print(predictions.shape)  # Output should match the input size (batch_size, height, width)


