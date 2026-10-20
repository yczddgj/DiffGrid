import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from datatools.feature_extractor.registry import register_extractor


@register_extractor('dino')
class DINOExtractor(nn.Module):
    def __init__(self, model_name: str = "dinov2_vits14", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device

        # 加载 Facebook DINOv2 模型
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval().to(device)

        # DINOv2 模型默认输出维度
        self.out_dim = self.model.embed_dim

        # 预处理（使用 DINOv2 官方推荐）
        self.preprocess_fn = transforms.Compose([
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def preprocess(self, pil_images: list[Image.Image]) -> torch.Tensor:
        tensors = [self.preprocess_fn(img).unsqueeze(0) for img in pil_images]
        return torch.cat(tensors, dim=0)  # [B, 3, 224, 224]

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x).float()
