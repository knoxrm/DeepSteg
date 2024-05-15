from efficientnet_pytorch import EfficientNet
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.vision_transformer import VisionTransformer
from transformers import ViTModel, ViTFeatureExtractor
from einops import rearrange
import timm
from surgery import *


class CustomReshapeLayer(nn.Module):

    def __init__(self, hidden_size, patch_size, image_size):
        super(CustomReshapeLayer, self).__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        batch_size, _, _, _ = x.shape

        # Reshape into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, self.num_patches, -1)

        # Linear projection to hidden_size
        projection_matrix = nn.Linear(x.size(-1), self.hidden_size)
        x = projection_matrix(x)

        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels):
        super(ProjectionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class EfficientNetCvT(nn.Module):
    def __init__(self, backbone):
        super(EfficientNetCvT, self).__init__()
        self.backbone = backbone
        feature_sizes = {
            'efficientnet-b0': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2304,
            # Add feature sizes for mixnet models if needed
        }
        feature_size = feature_sizes[backbone.model_name]

        # Vision Transformer
        hidden_dim = 512
        self.patch_size = 1
        self.patch_embed = nn.Linear(1280, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 16 * 16 + 1, hidden_dim))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
            for _ in range(6)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

        # Calculate the number of patches
        # self.image_size = 16 
        # self.patch_size = 4  # Adjust as needed
        #
        # out_channels = 144
        # self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        # # self.cross_projection = nn.MultiheadAttention(embed_dim=feature_size, kdim=768, vdim=768, num_heads=8).to("cuda")
        # self.cross_projection = nn.MultiheadAttention(embed_dim=768, num_heads=8).to("cuda")
        # self.query_projection = nn.Linear(feature_size, 768).to("cuda")
        # self.key_value_projection = nn.Linear(768, 768).to("cuda")
        # Define the CvT component
        # self.Vit = VisionTransformer(
        #     image_size=self.image_size,
        #     patch_size=self.patch_size,
        #     num_layers=4,
        #     num_heads=4,
        #     hidden_dim=feature_size,
        #     mlp_dim=4 * feature_size,
        #     dropout=0.1,
        #     attention_dropout=0.1,
        #     num_classes=4,
        #     representation_size=None,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # Add conv_stem_configs if required
        # )
        # self.conv = nn.Conv2d(2304, out_channels, kernel_size=4, stride=1, padding=0)
        # self.classifier = nn.Linear(2304, 4) 
        # self.linear_projection = nn.Linear(1280 + 768, 768)  # Adjust dimensions as needed
        # self.adapt_conv = ProjectionHead(2304,1152, 3)
        # self.adapt_conv = nn.Conv2d(2304, 3, kernel_size=1, stride=1, padding=0)
        # self.final_conv = nn.Conv2d(144, 4, kernel_size=1, stride=1, padding=0)
        


    def forward(self, x):
        features = self.backbone.extract_features(x)
        # features = self.efficientnet(x)
        patches = rearrange(features, 'b c h w -> b (h w) c')
        patches = self.patch_embed(patches)
        cls_token = self.cls_token.expand(patches.shape[0], -1, -1)
        patches = torch.cat((cls_token, patches), dim=1)
        patches += self.pos_embed
        
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        patches = self.norm(patches)
        cls_token_final = patches[:, 0]
        output = self.fc(cls_token_final)
        
        return output
    #
    # def forward(self, x):
    #     # x = self.backbone.extract_features(x)
    #     # x = x.mean(dim=[2, 3])  # Now shape is [batch_size, 2304]
    #     # x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
    #     # print(x.shape)
    #     # x = self.adapt_conv(x)
    #     # print(x.shape)
    #     # x = self.final_conv(x)
    #     # print(x.shape)
    #     # x = self.Vit(x)
    #     # Preprocess the input images
    #     x = (x - x.min()) / (x.max() - x.min()) 
    #     x = x.to("cuda")
    #     # x = x * 255.0 
    #
    #
    #     efficientnet_features = self.backbone.extract_features(x)
    #     B, C, H, W = efficientnet_features.shape
    #     queries = efficientnet_features.flatten(2).transpose(1,2)
    #     queries = self.query_projection(queries)
    #
    #     # vit_input = self.vit.patch_embed(x)
    #     vit_input = self.feature_extractor(images=x, return_tensors="pt", do_rescale=False).pixel_values
    #     vit_input = vit_input.to("cuda")
    #     keys_values = vit_input.flatten(2).transpose(1, 2)
    #     print("Shape of keys_values before projection:", keys_values.shape)
    #     keys_values = self.key_value_projection(keys_values)
    #     keys_values = keys_values.reshape(B, 3, -1)
    #     print(vit_input.shape)
    #     keys_values = vit_input.flatten(2).transpose(1, 2)
    #
    #     output, _ = self.cross_projection(queries, keys_values, keys_values)  
    #     # output = output.transpose(1, 2).reshape(B, N, C) 
    #     output = output.transpose(1, 2).reshape(B, -1, 768)
    #
    #     # Continue through ViT layers...
    #     output = self.vit.forward_features(output)
    #     # x = self.vit(x)
    #
    #     # x = self.classifier(x)
    #
    #     return output


def get_net(model_name, surgery):
    
    params = {
        'efficientnet-b0': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=1280, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b0')
        },
        'efficientnet-b2': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=1408, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b2')
        },
        
        'efficientnet-b4': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=1792, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b4')
        },
        
        'efficientnet-b5': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=2048, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b5')
        },
        
        'efficientnet-b6': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=2304, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b6')
        },

        'efficientnet-b7': {
            'fc_name': '_fc',
            'fc': nn.Linear(in_features=2304, out_features=2, bias=True),
            'init_op': partial(EfficientNet.from_pretrained, 'efficientnet-b7')
        },
        
        'mixnet_xl': {
            'fc_name': 'classifier',
            'fc': nn.Linear(in_features=1536, out_features=2, bias=True),
            'init_op': partial(timm.create_model, 'mixnet_xl', pretrained=True)
        },
            
        'mixnet_s': {
            'fc_name': 'classifier',
            'fc': nn.Linear(in_features=1536, out_features=2, bias=True),
            'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=True)
        }, 
    
        'mixnet_s_fromscratch': {
            'fc_name': 'classifier',
            'fc': nn.Linear(in_features=1536, out_features=2, bias=True),
            'init_op':  partial(timm.create_model, 'mixnet_s', pretrained=False)
        }, 
    }
    
    net = params[model_name]['init_op']()
    # setattr(net, params[model_name]['fc_name'], params[model_name]['fc'])
    setattr(net, params[model_name]['fc_name'], nn.Identity())  # Replace fc with Identity
    if surgery == 2:
        net = to_InPlaceABN(net)
        source = 'timm' if model_name.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)
    elif surgery == 1:
        source = 'timm' if model_name.startswith('mixnet') else 'efficientnet-pytorch'
        net = to_MishME(net, source=source)
    elif surgery == 3:
        net = remove_stride(net)
        net = add_pooling(net)
    elif surgery == 4:
        net = add_pooling(net)
    net.model_name = model_name
    model = EfficientNetCvT(backbone=net)
    return model
