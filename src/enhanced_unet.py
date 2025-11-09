import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp # type: ignore
from .se_blocks import SEBlock


class InstanceSegmentationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 3, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


class EnhancedUNet(nn.Module):
    def __init__(self, encoder_name='efficientnet-b7', num_classes=6):
        super().__init__()
        
        self.backbone = smp.Unet(
            encoder_name, 
            encoder_weights='imagenet', 
            classes=num_classes
        )
        
        # Get actual decoder output channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 96, 96)
            encoder_features = self.backbone.encoder(dummy_input)
            decoder_features = self.backbone.decoder(encoder_features)
            
            # The decoder returns a single tensor, not a list
            if isinstance(decoder_features, torch.Tensor):
                # Single SE block for the decoder output
                self.se_blocks = nn.ModuleList([
                    SEBlock(decoder_features.shape[1])
                ])
                self.instance_head = InstanceSegmentationHead(decoder_features.shape[1])
            else:
                # Multiple features (fallback)
                self.se_blocks = nn.ModuleList([
                    SEBlock(feat.shape[1]) for feat in decoder_features
                ])
                self.instance_head = InstanceSegmentationHead(decoder_features[-1].shape[1])
        
    def forward(self, x):
        encoder_features = self.backbone.encoder(x)
        decoder_features = self.backbone.decoder(encoder_features)
        
        # Handle single tensor output
        if isinstance(decoder_features, torch.Tensor):
            enhanced_feature = self.se_blocks[0](decoder_features)
            main_output = self.backbone.segmentation_head(enhanced_feature)
            instance_output = self.instance_head(enhanced_feature)
        else:
            # Handle multiple features (fallback)
            enhanced_features = []
            for feature, se_block in zip(decoder_features, self.se_blocks):
                enhanced_feature = se_block(feature)
                enhanced_features.append(enhanced_feature)
            
            main_output = self.backbone.segmentation_head(enhanced_features)
            instance_output = self.instance_head(enhanced_features[-1])
        
        return main_output, instance_output