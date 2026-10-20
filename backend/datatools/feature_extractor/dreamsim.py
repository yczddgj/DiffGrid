import torch
import torch.nn as nn
from datatools.feature_extractor.registry import register_extractor
from dreamsim import dreamsim


@register_extractor('dreamsim')
class DreamSimExtractor(nn.Module):
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        if dreamsim is None:
            raise ImportError("请先安装 dreamsim：pip install dreamsim")

        model, preprocess = dreamsim(pretrained=True)
        self.model = model.eval().to(device)
        self.preprocess_fn = preprocess  # 用于预处理 PIL.Image
        self.device = device

        # DreamSim 默认输出维度为 768
        self.out_dim = self.model.embed_size

    def preprocess(self, pil_images) -> torch.Tensor:
        # 将多个 PIL.Image 预处理为 [B, 3, 224, 224]
        tensors = [self.preprocess_fn(img) for img in pil_images]
        return torch.cat(tensors, dim=0)  # 每个预处理后的 tensor 是 [1, 3, 224, 224]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [B, 3, 224, 224]，预处理后的图像张量
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model.embed(x).float()
