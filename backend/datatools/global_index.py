import os
import json
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
from bisect import bisect_right
from config import INDEX_DIR, DATASET_DIR
from datatools.feature_extractor.runner import extract_features_from_slice

_global_index_instance = None


def get_global_index():
    global _global_index_instance
    if _global_index_instance is None:
        _global_index_instance = GlobalIndex()
    return _global_index_instance


class GlobalIndex:
    _instance = None  # 类变量，保存全局唯一实例

    def __new__(cls, index_dir=None, dataset_dir=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, index_dir: str = INDEX_DIR, dataset_dir: str = DATASET_DIR):
        if self._initialized:
            return  # 避免重复初始化
        self.index_dir = index_dir
        self.dataset_dir = dataset_dir
        self.global_index_path = os.path.join(index_dir, 'global_index.json')
        self.blocks_dir = os.path.join(index_dir, 'blocks')
        self.global_index = self._load_global_index()
        self.block_cache = {}
        self.image_list_cache = {}
        self.np_cache = {}
        self._initialized = True

    def reload(self):
        """从磁盘重载 index 信息"""
        self.global_index = self._load_global_index()
        self.block_cache.clear()
        self.image_list_cache.clear()
        self.np_cache.clear()

    def _load_global_index(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.global_index_path):
            return []
        with open(self.global_index_path, 'r') as f:
            return json.load(f)

    def _find_block(self, global_id: int) -> Dict[str, Any]:
        starts = [entry['start_id'] for entry in self.global_index]
        idx = bisect_right(starts, global_id) - 1
        if idx < 0:
            raise ValueError(f'global_id {global_id} 太小，不在任何 block 中')
        entry = self.global_index[idx]
        if global_id >= entry['start_id'] + entry['length']:
            raise ValueError(f'global_id {global_id} 超出 block 范围')
        return entry

    def _load_block(self, block_path: str) -> Dict[str, Any]:
        if block_path not in self.block_cache:
            full_path = os.path.join(self.index_dir, block_path)
            with open(full_path, 'r') as f:
                self.block_cache[block_path] = json.load(f)
        return self.block_cache[block_path]

    def _load_npy(self, rel_path: str) -> np.ndarray:
        if rel_path not in self.np_cache:
            full_path = os.path.join(self.dataset_dir, rel_path)
            self.np_cache[rel_path] = np.load(full_path, mmap_mode='r')
        return self.np_cache[rel_path]

    def _load_image_list(self, block_path: str, image_index_path: str, img_type='img') -> List[str]:
        if block_path + img_type not in self.image_list_cache:
            full_path = os.path.join(self.dataset_dir, image_index_path)
            with open(full_path, 'r') as f:
                self.image_list_cache[block_path + img_type] = json.load(f)
        return self.image_list_cache[block_path + img_type]

    def get_info(self, global_id: int) -> Dict[str, Any]:
        block_meta = self._find_block(global_id)
        block = self._load_block(block_meta['block_path'])
        offset = global_id - block_meta['start_id']
        return {
            'block_meta': block_meta,
            'block': block,
            'offset': offset
        }

    def get_infos(self, global_ids: List[int]) -> List[Dict[str, Any]]:
        return [self.get_info(gid) for gid in global_ids]

    def get_thumbnail_ids(self, global_ids: List[int]):
        # 假设所有 global_ids 属于同一个 block
        block_meta = self._find_block(global_ids[0])
        block_path = block_meta['block_path']
        offsets = np.array(global_ids) - block_meta['start_id']

        block = self._load_block(block_path)
        thumbnail_index_path = block.get('thumbnail_index_path')
        if not thumbnail_index_path:
            return np.array(range(len(global_ids)), dtype=int)

        thumbnail_list = self._load_image_list(
            block_path, thumbnail_index_path, img_type='thumbnail'
        )
        # 转为 numpy 数组以便批量操作
        thumbnail_arr = np.array(thumbnail_list, dtype=object)
        selected_thumbs = thumbnail_arr[offsets]
        valid_mask = selected_thumbs != ''
        return np.where(valid_mask)[0]

    def get_title_ids(self, global_ids: List[int]):
        # 假设所有 global_ids 属于同一个 block
        block_meta = self._find_block(global_ids[0])
        block_path = block_meta['block_path']

        block = self._load_block(block_path)
        title_info = block.get('title', {})
        title_indices = title_info.get('indices_path')
        title_path = title_info.get('titles_path')
        if not title_path or not title_indices:
            print(f"No title path for block {block_path}")
            return self.get_thumbnail_ids(global_ids)
        indices_array = self._load_npy(title_indices)
        return indices_array

    def get_block_feature(self, gid, model_name=None) -> List[Any]:
        '''
        核心集获取全集的特征向量
        '''
        block_meta = self._find_block(gid)
        block_path = block_meta['block_path']

        block = self._load_block(block_path)
        features_dict = block.get("features", {})

        if not features_dict:
            raise ValueError(
                f"[×] Block {block_path} 中缺少 features 字段")

        # 如果未指定模型，则默认选择一个
        if model_name == None:
            model_name = next(iter(features_dict.keys()))

        # 若模型不存在，动态提取
        if model_name not in features_dict:
            print(f"[+] 模型 {model_name} 特征缺失，正在提取: {block_path}")

            # 构建 Slice（不保存）
            start_id = block_meta['start_id']
            length = block_meta['length']
            gids = list(range(start_id, start_id + length))
            from datatools.slice import Slice
            slice_temp = Slice(
                name=f'temp-{start_id}', global_ids=gids, overwrite=True)

            # 构造保存路径（基于 image_index 同目录）
            image_index_path = block.get("image_index_path")
            if not image_index_path:
                raise ValueError(
                    f"No image_index_path in block {block_path}")
            index_dir = os.path.dirname(os.path.join(
                self.dataset_dir, image_index_path))
            output_name = os.path.splitext(os.path.basename(image_index_path))[
                0] + f'_{model_name}.npy'
            output_path = os.path.join(index_dir, output_name)
            rel_output_path = os.path.relpath(
                output_path, self.dataset_dir)

            # 提取特征
            extract_features_from_slice(
                slice_temp, output_path, extractor_name=model_name
            )

            # 更新 block 文件
            features_dict[model_name] = rel_output_path
            block["features"] = features_dict

            full_block_path = os.path.join(
                self.index_dir, block_path)
            with open(full_block_path, 'w') as f:
                json.dump(block, f, indent=2)
            self.block_cache[block_path] = block

        # 加载特征数组
        feature_arrays = self._load_npy(block["features"][model_name])
        return feature_arrays

    def get_field_by_ids(self, global_ids: List[int], field: str, model_name=None) -> List[Any]:
        block_to_ids = defaultdict(list)
        for gid in global_ids:
            block_meta = self._find_block(gid)
            block_path = block_meta['block_path']
            offset = gid - block_meta['start_id']
            block_to_ids[block_path].append((gid, offset))

        results = {}

        for block_path, gid_offset_list in block_to_ids.items():
            block = self._load_block(block_path)

            if field == 'image_path':
                image_index_path = block.get('image_index_path')
                if not image_index_path:
                    raise ValueError(f'Block {block_path} 中没有 image_index_path')
                image_list = self._load_image_list(block_path, image_index_path)
                image_index_dir = os.path.dirname(
                    os.path.join(self.dataset_dir, image_index_path))
                for gid, offset in gid_offset_list:
                    results[gid] = os.path.normpath(
                        os.path.join(image_index_dir, image_list[offset]))

            elif field == 'thumbnail_path':
                thumbnail_index_path = block.get('thumbnail_index_path')

                if not thumbnail_index_path:
                    return [''] * len(global_ids)  # 如果没有缩略图索引，返回空列表
                    # print(f"[+] Block {block_path} 缺少 thumbnail，正在生成...")

                    # # 1. 构建 Slice
                    # block_meta = next(
                    #     b for b in self.global_index if b['block_path'] == block_path)
                    # start_id = block_meta['start_id']
                    # length = block_meta['length']
                    # gids = list(range(start_id, start_id + length))
                    # from datatools.slice import Slice
                    # slice_temp = Slice(name=f'temp-{start_id}',
                    #                 global_ids=gids, overwrite=True)

                    # # 2. 构造输出路径
                    # image_index_path = block.get('image_index_path')
                    # if not image_index_path:
                    #     raise ValueError(f'Block {block_path} 中没有 image_index_path')
                    # index_dir = os.path.dirname(os.path.join(
                    #     self.dataset_dir, image_index_path))
                    # thumbnail_index_path = os.path.join(
                    #     index_dir, f"thumbnail_{start_id}.json")
                    # rel_thumbnail_index_path = os.path.relpath(
                    #     thumbnail_index_path, self.dataset_dir)

                    # # 3. 调用 thumbnail 生成函数（你后续实现）
                    # generate_thumbnails_with_clip(slice_temp, output_dir=index_dir)

                    # # 4. 更新 block 文件
                    # block['thumbnail_index_path'] = rel_thumbnail_index_path
                    # full_block_path = os.path.join(self.index_dir, block_path)
                    # with open(full_block_path, 'w') as f:
                    #     json.dump(block, f, indent=2)
                    # self.block_cache[block_path] = block
                    # print(f"[✓] Block {block_path} 缺失 thumbnail 已生成")

                # 5. 加载 thumbnail 文件并返回路径
                thumbnail_list = self._load_image_list(
                    block_path, thumbnail_index_path, img_type='thumbnail')

                thumbnail_index_dir = os.path.dirname(
                    os.path.join(self.dataset_dir, thumbnail_index_path))
                for gid, offset in gid_offset_list:
                    results[gid] = os.path.normpath(
                        os.path.join(thumbnail_index_dir,
                                     thumbnail_list[offset])
                    ) if thumbnail_list[offset] != '' else ''
                    # print(f"获取缩略图路径: {results[gid]},", thumbnail_list[offset])

            elif field == 'embedding':
                npy_info = block.get(field, {})
                npy_path = npy_info.get('path')
                if not npy_path:
                    raise ValueError(f"No {field} path in block {block_path}")
                npy_array = self._load_npy(npy_path)
                for gid, offset in gid_offset_list:
                    results[gid] = npy_array[offset]

            elif field == 'feature':
                # 支持 model_name 为 str 或 list[str]
                model_names = model_name if isinstance(
                    model_name, list) else [model_name]

                for block_path, gid_offset_list in block_to_ids.items():
                    block = self._load_block(block_path)
                    features_dict = block.get("features", {})

                    if not features_dict:
                        raise ValueError(
                            f"[×] Block {block_path} 中缺少 features 字段")

                    # 如果未指定模型，则默认选择一个
                    if model_names == [None]:
                        model_names = [next(iter(features_dict.keys()))]

                    # 若部分模型不存在，动态提取
                    for m in model_names:
                        if m not in features_dict:
                            print(f"[+] 模型 {m} 特征缺失，正在提取: {block_path}")

                            # 构建 Slice（不保存）
                            block_meta = next(
                                b for b in self.global_index if b['block_path'] == block_path)
                            start_id = block_meta['start_id']
                            length = block_meta['length']
                            gids = list(range(start_id, start_id + length))
                            from datatools.slice import Slice
                            slice_temp = Slice(
                                name=f'temp-{start_id}', global_ids=gids, overwrite=True)

                            # 构造保存路径（基于 image_index 同目录）
                            image_index_path = block.get("image_index_path")
                            if not image_index_path:
                                raise ValueError(
                                    f"No image_index_path in block {block_path}")
                            index_dir = os.path.dirname(os.path.join(
                                self.dataset_dir, image_index_path))
                            output_name = os.path.splitext(os.path.basename(image_index_path))[
                                0] + f'_{m}.npy'
                            output_path = os.path.join(index_dir, output_name)
                            rel_output_path = os.path.relpath(
                                output_path, self.dataset_dir)

                            # 提取特征
                            extract_features_from_slice(
                                slice_temp, output_path, extractor_name=m
                            )

                            # 更新 block 文件
                            features_dict[m] = rel_output_path
                            block["features"] = features_dict

                            full_block_path = os.path.join(
                                self.index_dir, block_path)
                            with open(full_block_path, 'w') as f:
                                json.dump(block, f, indent=2)
                            self.block_cache[block_path] = block

                    # 加载所有特征数组（按模型名顺序拼接）
                    feature_arrays = {}
                    for m in model_names:
                        npy_path = block["features"][m]
                        feature_arrays[m] = self._load_npy(npy_path)

                    # 按照每个 global_id 拼接特征
                    for gid, offset in gid_offset_list:
                        feats = [feature_arrays[m][offset] for m in model_names]
                        results[gid] = np.concatenate(feats, axis=-1)

                return [results[gid] for gid in global_ids]

            elif field == 'label':
                label_info = block.get('label', {})
                label_type = label_info.get('type')
                label_path = label_info.get('path')
                if not label_path:
                    raise ValueError(f"No label path for block {block_path}")

                if label_type == 'npy':
                    label_array = self._load_npy(label_path)
                    for gid, offset in gid_offset_list:
                        results[gid] = label_array[offset]
                elif label_type == 'json':
                    return [0]
                else:
                    raise ValueError(
                        f"Unsupported label type '{label_type}' in block {block_path}")

            elif field == 'title':
                title_info = block.get('title', {})
                title_indices = title_info.get('indices_path')
                title_path = title_info.get('titles_path')
                if not title_path or not title_indices:
                    print(f"No title path for block {block_path}")
                    results[gid] = "demo title"
                    continue
                # indices和title_list长度一致，前者保存block中有title的图片索引，后者为对应的title
                indices_array = self._load_npy(title_indices)
                if offset not in indices_array:
                    results[gid] = ""
                    print(f"Global ID {gid} 的标题索引 {offset} 不在有效范围内")
                else:
                    # 获取标题列表
                    title_list = self._load_image_list(
                        block_path, title_path, img_type='title')
                    # print(f"标题列表长度: {len(title_list)}", offset)
                    results[gid] = title_list[indices_array.tolist().index(offset)]
                    # print(f"获得标题: {results[gid]}")

            elif field == 'key_word':
                title_info = block.get('title', {})
                title_indices = title_info.get('indices_path')
                title_path = title_info.get('key_word_path')
                if not title_path or not title_indices:
                    print(f"No key word path for block {block_path}")
                    results[gid] = "demo title"
                    title_path = title_info.get('titles_path')
                    if not title_path or not title_indices:
                        print(f"No title path for block {block_path}")
                        results[gid] = "demo title"
                        continue
                    # indices和title_list长度一致，前者保存block中有title的图片索引，后者为对应的title
                    indices_array = self._load_npy(title_indices)
                    if offset not in indices_array:
                        results[gid] = ""
                        print(f"Global ID {gid} 的标题索引 {offset} 不在有效范围内")
                    else:
                        # 获取标题列表
                        title_list = self._load_image_list(
                            block_path, title_path, img_type='title')
                        # print(f"标题列表长度: {len(title_list)}", offset)
                        results[gid] = title_list[indices_array.tolist().index(offset)]
                        # print(f"获得标题: {results[gid]}")
                else:
                    # indices和title_list长度一致，前者保存block中有title的图片索引，后者为对应的title
                    indices_array = self._load_npy(title_indices)
                    if offset not in indices_array:
                        results[gid] = ""
                        print(f"Global ID {gid} 的标题索引 {offset} 不在有效范围内")
                    else:
                        # 获取标题列表
                        title_list = self._load_image_list(
                            block_path, title_path, img_type='key_word')
                        # print(f"key_word长度: {len(title_list)}", offset)
                        results[gid] = title_list[indices_array.tolist().index(offset)]
                        # print(f"获得key word: {results[gid]}")

            else:
                raise ValueError(f'不支持的字段名：{field}')

        return [results[gid] for gid in global_ids]


if __name__ == "__main__":
    gi = GlobalIndex()
    embeddings = gi.get_field_by_ids([10001, 10002, 10003], 'embedding')
    labels = gi.get_field_by_ids([10001, 10002], 'label')
