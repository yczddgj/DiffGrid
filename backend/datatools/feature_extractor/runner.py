import os
from torch.utils.data import DataLoader
import json
import numpy as np
from tqdm import tqdm
import torch
from datatools.feature_extractor.registry import get_extractor



def extract_features_from_slice(
    slice_obj,
    output_path,
    extractor_name='resnet50',
    batch_size=32,
    num_workers=4,
    device=None
):
    """
    使用 DataLoader 和 GPU 提取 Slice 中所有图像的特征

    参数：
    - slice_obj: 一个已初始化的 Slice 对象，必须继承 Dataset
    - output_path: 保存特征的 .npy 路径
    - extractor_name: 使用的特征提取器名（需注册在 EXTRACTOR_REGISTRY 中）
    - batch_size: 推理 batch 大小
    - num_workers: DataLoader 加载线程数
    - device: GPU 设备（默认自动选择）

    生成：
    - output_path: 提取后的特征数组
    - output_path.replace('.npy', '_index.json'): 对应的 image_path 列表
    """
    device = torch.device(device or (
        'cuda' if torch.cuda.is_available() else 'cpu'))
    extractor_cls = get_extractor(extractor_name)
    if extractor_cls is None:
        raise ValueError(f'Unknown feature extractor: {extractor_name}')
    extractor = extractor_cls().to(device).eval()

    loader = DataLoader(slice_obj, batch_size=batch_size,
                        num_workers=num_workers)

    features = []
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(loader):
            batch_imgs = batch_imgs.to(device)
            feats = extractor(batch_imgs)  # 输出 [B, D]
            feats = feats.cpu().numpy()
            features.append(feats)

    features = np.concatenate(features, axis=0)
    np.save(output_path, features)

    print(f"[✓] 提取完成：共 {len(features)} 张图像特征，保存至 {output_path}")
