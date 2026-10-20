from torch.utils.data import Dataset
from typing import List, Any
import cv2
import os
import numpy as np
from typing import List, Optional, Union, Callable, Any, Tuple, Dict
import json
from PIL import Image
from .global_index import get_global_index
from config import INDEX_DIR, DATASET_DIR, SLICE_DIR
import config

import torch
import numpy as np
from torchvision import transforms


class CV2ToTensor:
    def __call__(self, image):
        # image: np.ndarray, HWC, RGB, uint8
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
        return image


transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.resize(img, (224, 224))),  # <== 统一尺寸
    CV2ToTensor()
])


class Slice(Dataset):
    def __init__(
        self,
        name: str,
        global_ids: Optional[Union[List[int], np.ndarray]] = None,
        overwrite: bool = False,
        transform: Optional[Callable] = transform,
        target_transform: Optional[Callable] = None,
    ):
        """
        - name: slice 名（不带后缀）
        - global_ids: 提供则新建；否则 load
        - overwrite: 是否覆盖已存在同名 slice
        - transform: 应用于 image 的函数 (PIL->...)
        - target_transform: 应用于 label 的函数
        """
        self.name = name
        if config.MODE == "coreset":
            self.slice_path = os.path.join(SLICE_DIR, "coreset", f'{name}.npy')
        else:
            self.slice_path = os.path.join(SLICE_DIR, f'{name}.npy')
        self.global_index = get_global_index()
        self.transform = transform
        self.target_transform = target_transform

        self.classes = []

        # 初始化 global_ids
        if global_ids is not None:
            if os.path.exists(self.slice_path) and not overwrite:
                raise FileExistsError(
                    f"Slice '{self.name}' 已存在，若要覆盖请设置 overwrite=True"
                )
            self.global_ids = np.array(global_ids, dtype=np.int64)
        else:
            self.global_ids = self.load()

    def save(self, sort: bool = True):
        """保存为 .npy 文件，默认自动排序去重"""
        if config.MODE == "coreset":
            coreset_dir = os.path.join(SLICE_DIR, "coreset")
            os.makedirs(coreset_dir, exist_ok=True)
        else:
            os.makedirs(SLICE_DIR, exist_ok=True)
        if sort:
            self.global_ids = np.unique(np.sort(self.global_ids))
        np.save(self.slice_path, self.global_ids)

    def save_subset(self, subset_name, subset: np.ndarray, sort: bool = True):
        """保存为 .npy 文件，默认自动排序去重"""
        subset_ids = self.global_ids[subset]
        if sort:
            subset_ids = np.unique(np.sort(subset_ids))
        subset_slice = Slice(subset_name, global_ids=subset_ids, overwrite=True)
        subset_slice.save()
        return subset_slice

    def load(self) -> np.ndarray:
        """从 .npy 文件加载，并保证唯一有序"""
        if not os.path.exists(self.slice_path):
            raise FileNotFoundError(f'Slice 文件不存在: {self.slice_path}')
        arr = np.load(self.slice_path)
        return np.unique(np.sort(arr))

    @property
    def is_small(self) -> bool:
        # Todo 需要修改对当前图像大小的判断
        return True

    def __len__(self) -> int:
        return len(self.global_ids)

    def __repr__(self):
        return f"<Slice {self.name} | size={len(self)}>"

    @staticmethod
    def list_all() -> List[str]:
        """返回所有已存在的 slice 名称"""
        if config.MODE == "coreset":
            coreset_dir=os.path.join(SLICE_DIR, "coreset")
            if not os.path.exists(coreset_dir):
                return []
            return sorted([
                f[:-4]
                for f in os.listdir(coreset_dir)
                if f.endswith('.npy')
            ])
        else:
            if not os.path.exists(SLICE_DIR):
                return []
            return sorted([
                f[:-4]
                for f in os.listdir(SLICE_DIR)
                if f.endswith('.npy')
            ])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Dataset 标准接口：返回单个样本 (image, label)
        """
        return self.get_sample(idx)

    def _get_samples_by_indices(self, indices: List[int]) -> List[Tuple[Any, Any]]:
        """
        批量获取多个样本（加速路径）
        """
        gids = self.global_ids[indices].tolist() if isinstance(
            indices, np.ndarray) else [self.global_ids[i] for i in indices]

        # 批量读取 image_path 和 label
        image_paths = self.global_index.get_field_by_ids(gids, 'image_path')
        labels = self.global_index.get_field_by_ids(gids, 'label')

        samples = []
        for img_path, label in zip(image_paths, labels):
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            samples.append((image, label))

        return samples

    def get_block_info(self) -> dict:
        """
        返回当前 Slice 所属 block 的元信息（从 global_index 中查询第一个 global_id 所在 block）
        """
        return self.global_index.get_info(self.global_ids[0])['block']

    def get_global_id(self, idx: int) -> int:
        """返回第 idx 个样本的 global_id"""
        return int(self.global_ids[idx])

    def get_block_offset(self):
        """获取当前 Slice 在所属 block 中的偏移量"""
        start_id = self.global_index.get_info(self.global_ids[0])[
            'block_meta']['start_id']
        return np.array(self.global_ids - start_id)

    def get_image_path(self, idx: int) -> str:
        """快速获取第 idx 个样本的图像路径"""
        gid = self.get_global_id(idx)
        return self.global_index.get_field_by_ids([gid], 'image_path')[0]

    def get_thumbnail_path(self, idx: int) -> str:
        """快速获取第 idx 个样本的缩略图路径"""
        gid = self.get_global_id(idx)
        return self.global_index.get_field_by_ids([gid], 'thumbnail_path')[0]

    def get_thumbnail_paths(self, indices: List[int]) -> List[str]:
        """快速获取多个样本的缩略图路径"""
        gids = self.global_ids[indices].tolist()
        return self.global_index.get_field_by_ids(gids, 'thumbnail_path')


    def get_label(self, idx: int) -> Any:
        """快速获取第 idx 个样本的原始 label（取决于 block 配置）"""
        gid = self.get_global_id(idx)
        return self.global_index.get_field_by_ids([gid], 'label')[0]

    def get_image(self, idx: int) -> Any:
        """加载并返回图像，应用 transform（若有）。若出错则返回黑图并打印警告"""
        path = self.get_image_path(idx)
        try:
            image = cv2.imread(path)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[×] 加载图像失败：{path} | 错误信息：{e}")
            # 返回黑图（默认 224x224x3），可根据你的模型输入大小调整
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            try:
                img = self.transform(img)
            except Exception as e:
                print(f"[×] 图像 transform 失败：{path} | 错误信息：{e}")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                if self.transform:
                    try:
                        img = self.transform(img)
                    except:
                        pass  # 若黑图也失败则直接返回

        return img

    def get_target(self, idx: int) -> Any:
        """获取并转换 label"""
        lbl = self.get_label(idx)
        if self.target_transform:
            lbl = self.target_transform(lbl)
        return lbl

    def get_sample(self, idx: int) -> Tuple[Any, Any]:
        """
        返回 (image, label) 二元组，可视为 Dataset[idx]
        """
        return self.get_image(idx), self.get_target(idx)

    # —— 批量字段提取加速 —— #
    def get_image_paths(self, indices: List[int]) -> List[str]:
        """
        批量提取该 slice 所有样本的某个字段，如 'embedding','feature','image_path','label'
        """
        gids = self.global_ids[indices].tolist()
        return self.global_index.get_field_by_ids(gids, 'image_path')

    def get_title(self, idx: int) -> Optional[str]:
        """获取第 idx 个样本的图表标题"""
        gid = self.get_global_id(idx)
        return self.global_index.get_field_by_ids([gid], 'title')[0]

    def get_key_word(self, idx: int) -> Optional[str]:
        """获取第 idx 个样本的图表标题"""
        gid = self.get_global_id(idx)
        return self.global_index.get_field_by_ids([gid], 'key_word')[0]

    def get_field(self, field: str, feature_type=None) -> List[Any]:
        """
        批量提取该 slice 所有样本的某个字段，如 'embedding','feature','image_path','label'
        """
        gids = self.global_ids.tolist()
        return self.global_index.get_field_by_ids(gids, field, feature_type)

    def get_block_feature(self, feature_type) -> Any:
        """快速获取block全部特征"""
        gid = self.global_ids[0]
        return self.global_index.get_block_feature(gid, feature_type)

    def get_father_slice(self):
        """获取父级切片"""
        block_meta = self.global_index.get_info(self.global_ids[0])[
            'block_meta']
        start_id = block_meta['start_id']
        length = block_meta['length']
        gids = list(range(start_id, start_id + length))
        slice_temp = Slice(
            name=f'temp-{start_id}', global_ids=gids)
        return slice_temp

    def get_title_indices(self):
        """获取所有缩略图路径不为空的样本索引列表"""
        gids = self.global_ids.tolist()
        return self.global_index.get_title_ids(gids)

    def get_thumb_indices(self):
        """获取所有缩略图路径不为空的样本索引列表"""
        gids = self.global_ids.tolist()
        return self.global_index.get_thumbnail_ids(gids)

    def get_image_url(self, idx: int, base_url: str = "/static/datasets", thumbnail=False) -> str:
        """获取第 idx 个样本的Web可访问图片URL"""
        if thumbnail:
            image_path = self.get_thumbnail_path(idx)
            if image_path == '':
                return ''
        else:
            image_path = self.get_image_path(idx)
        return self._convert_path_to_url(image_path, base_url)

    def get_image_urls(self, indices: List[int], base_url: str = "/static/datasets", thumbnail=False) -> List[str]:
        """批量获取多个样本的Web可访问图片URL"""
        if not indices:
            return []

        # 批量获取图片路径
        gids = [self.global_ids[i] for i in indices]
        if thumbnail:
            image_paths = self.global_index.get_field_by_ids(
                gids, 'thumbnail_path')
            # 统计为空的缩略图id，替换为image_path
        else:
            image_paths = self.global_index.get_field_by_ids(gids, 'image_path')

        # 转换为URL
        return [self._convert_path_to_url(path, base_url) for path in image_paths]

    def _convert_path_to_url(self, image_path: str, base_url: str = "/static/datasets") -> str:
        """将图片路径转换为Web可访问的URL"""
        import os
        if image_path == '':
            return ''
        from urllib.parse import quote
        relative_path = os.path.relpath(image_path, DATASET_DIR)
        if relative_path == '..' or relative_path.startswith('..' + os.sep):
            raise ValueError('Image path must be inside DIFFGRID_DATA_DIR/datasets')
        return f"{base_url}/{quote(relative_path.replace(os.sep, '/'))}"



def generate_slices_from_dataset(dataset_name: str, dataset_root: str = DATASET_DIR, index_path: str = INDEX_DIR, save_dir: str = SLICE_DIR):
    """
    根据 info.json 和 index.json 生成标准化切片
    """
    dataset_dir = os.path.join(dataset_root, dataset_name)
    info_path = os.path.join(dataset_dir, 'info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"No info.json found in {dataset_dir}")

    with open(info_path, 'r') as f:
        info = json.load(f)

    index_file = os.path.join(index_path, "global_index.json")
    with open(index_file, 'r') as f:
        index = json.load(f)

    os.makedirs(save_dir, exist_ok=True)

    if info['task'] == 'detection':
        _generate_detection_slices(info, index, save_dir)
    elif info['task'] == 'classification':
        _generate_classification_slices(
            dataset_name, info, index, dataset_dir, save_dir)
    else:
        raise ValueError(f"Unknown task type: {info['task']}")


def _generate_detection_slices(info: Dict, index: List[Dict], save_dir: str):
    """
    生成 detection 类型数据集的切片
    """
    type_to_start_id = {}
    for t in info['types']:
        matched = next((i for i in index if info['dataset_name'].lower(
        ) in i['discription'].lower() and t in i['discription'].lower()), None)
        if not matched:
            raise ValueError(
                f"No matching block found for type '{t}' in dataset '{info['dataset_name']}'")
        type_to_start_id[t] = matched['start_id']

    for t in info['types']:
        start_id = type_to_start_id[t]
        for split, value in info['splits'][t].items():
            begin, end = value['indices']
            global_ids = np.arange(
                start_id + begin, start_id + end + 1, dtype=np.int64)
            slice_name = f"{info['dataset_name']}-{t}-{split}"
            path = os.path.join(save_dir, f"{slice_name}.npy")
            np.save(path, np.unique(global_ids))
            print(
                f"[Detection] Saved slice: {slice_name}, count: {len(global_ids)}")


def _generate_classification_slices(dataset_name: str, info: Dict, index: List[Dict], dataset_dir: str, save_dir: str):
    """
    生成 classification 类型数据集的切片
    """
    matched = next((i for i in index if i['discription'].lower(
    ).startswith(dataset_name.lower())), None)
    if not matched:
        raise ValueError(
            f"No matching block found for classification dataset '{dataset_name}'")

    start_id = matched['start_id']
    for split, value in info['splits'].items():
        indices_file = os.path.join(dataset_dir, value['indices_file'])
        if not os.path.exists(indices_file):
            raise FileNotFoundError(f"Missing indices file: {indices_file}")
        rel_indices = np.load(indices_file)
        global_ids = rel_indices + start_id
        slice_name = f"{dataset_name}-{split}"
        path = os.path.join(save_dir, f"{slice_name}.npy")
        np.save(path, np.unique(global_ids))
        print(
            f"[Classification] Saved slice: {slice_name}, count: {len(global_ids)}")


def filter_slice_by_labels(
    source: str,
    include_labels: List[Any],
    new_name: str,
    overwrite: bool = False
) -> Slice:
    """
    从 source_slice 中筛选出标签在 include_labels 中的样本，生成新的 Slice。

    参数：
    - source_slice: 原始 Slice 实例
    - include_labels: 要保留的标签值列表
    - new_name: 新 Slice 的名称
    - overwrite: 是否覆盖已有的新 Slice 文件

    返回：
    - 新 Slice 实例
    """
    source_slice = Slice(source)
    # 获取标签列表
    labels = source_slice.get_field('label')

    # 获取符合条件的索引
    matched_indices = [i for i, lbl in enumerate(
        labels) if lbl in include_labels]

    # 提取对应 global_ids
    matched_global_ids = source_slice.global_ids[matched_indices]

    # 创建并保存新 Slice
    new_slice = Slice(
        name=new_name, global_ids=matched_global_ids, overwrite=overwrite)
    new_slice.save()

    return new_slice


if __name__ == '__main__':
    # generate_slices_from_dataset('EAGLE')     # 生成 EAGLE 的 scene/object 切片
    # generate_slices_from_dataset('CIFAR10')   # 生成 CIFAR10 的 train/val 切片
    s = Slice('EAGLE-scene-train')
    # s = Slice('CIFAR10-train')
    print(s)
    print(s[0:10])
    print(s.list_all())
