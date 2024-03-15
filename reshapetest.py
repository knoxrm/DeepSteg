import torch
import torch.nn as nn

class ReshapeForViT(nn.Module):
    def __init__(self, patch_size, num_channels, height, width):
        super(ReshapeForViT, self).__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, _, _, _ = x.shape

        # Reshape: Split channels into RGB groups
        x = x.view(batch_size, self.num_channels, 3, self.height, self.width)

        # Reshape into patches: [batch_size, num_patches, RGB, patch_height, patch_width]
        x = x.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, 3, self.patch_size, self.patch_size)

        return x

# Example usage
patch_size = 4    # Size of each patch
num_channels = 2304 // 3  # Channels per RGB group
height, width = 16, 16    # Height and width of the image

reshape_layer = ReshapeForViT(patch_size, num_channels, height, width)

# Dummy input
input_tensor = torch.randn(32, 2304, 16, 16)  # Replace with actual output from ImageNet model

# Reshape for ViT
output_tensor = reshape_layer(input_tensor)
print(output_tensor.shape)

