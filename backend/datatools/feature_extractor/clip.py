import clip
import torch
import torch.nn as nn
from torchvision import transforms
from datatools.feature_extractor.registry import register_extractor


@register_extractor('clip')
class CLIPExtractor(nn.Module):
    def __init__(self, model_name="ViT-B/32", device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device
        self.out_dim = self.model.visual.output_dim  # 通常是 512

    def forward(self, x: torch.Tensor):
        # x: tensor, [B, 3, H, W], 已转换为 clip.preprocess 格式
        with torch.no_grad():
            return self.model.encode_image(x).float()
