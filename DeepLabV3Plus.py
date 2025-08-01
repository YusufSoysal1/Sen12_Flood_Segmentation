import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.ReLU()

        )

        self.project = nn.Sequential(
            nn.Conv2d(256 * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=8, out_channels=1):
        super().__init__()

        # Backbone: ResNet50
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone = IntermediateLayerGetter(backbone, return_layers={
            'layer1': 'low_level',
            'layer4': 'out'
        })

        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Low-level feature projeksiyonu
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        low_level_feat = features['low_level']
        x = features['out']

        x = self.aspp(x)
        low_level_feat = self.low_level_proj(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x  # raw logits
