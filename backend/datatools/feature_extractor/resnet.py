import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from datatools.feature_extractor.registry import register_extractor


@register_extractor('resnet18')
class ResNet18Extractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # 去除 fc
        self.out_dim = model.fc.in_features  # e.g. 512

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


@register_extractor('resnet50')
class ResNet50Extractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = model.fc.in_features  # 2048

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)
