import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



class EAGLEImageDataset(Dataset):
    def __init__(self, root_dir='./datasets/EAGLE', transform=None, return_split=False):
        """
        :param json_index_path: 包含图像路径的 JSON 索引文件
        :param root_dir: 图像相对路径的根目录（可选）
        :param transform: torchvision.transforms，用于图像预处理
        :param return_split: 是否返回数据划分信息（Train/Val/Test）
        """
        self.root_dir = root_dir+'/EAGLE'
        self.transform = transform
        self.return_split = return_split
        json_index_path = os.path.join(self.root_dir, 'filenames.json')
        with open(json_index_path, 'r', encoding='utf-8') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        full_path = os.path.join(self.root_dir, path) if self.root_dir else path

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"[Error] 加载图像失败: {full_path}: {e}")
            # 返回空图像防止报错
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        if self.return_split:
            if 'Train' in path:
                split = 'Train'
            elif 'Val' in path:
                split = 'Val'
            elif 'Test' in path:
                split = 'Test'
            else:
                split = 'Unknown'
            return image, split, path

        return image


class SubImageDataset(Dataset):
    def __init__(self, root_dir='./datasets/EAGLE', transform=None, return_split=False):
        """
        :param json_index_path: 包含图像路径的 JSON 索引文件
        :param root_dir: 图像相对路径的根目录（可选）
        :param transform: torchvision.transforms，用于图像预处理
        :param return_split: 是否返回数据划分信息（Train/Val/Test）
        """
        self.root_dir = root_dir+'/EAGLE'
        self.transform = transform
        self.return_split = return_split
        json_index_path = os.path.join(self.root_dir, 'sub_filenames.json')
        with open(json_index_path, 'r', encoding='utf-8') as f:
            self.image_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        full_path = os.path.join(
            self.root_dir, rel_path) if self.root_dir else rel_path

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"[Error] 加载图像失败: {full_path}: {e}")
            image = Image.new('RGB', (100, 100))

        if self.transform:
            image = self.transform(image)

        if self.return_split:
            # 推断所属 split
            if 'Train' in rel_path:
                split = 'Train'
            elif 'Val' in rel_path:
                split = 'Val'
            elif 'Test' in rel_path:
                split = 'Test'
            else:
                split = 'Unknown'
            return image, split, rel_path

        return image
