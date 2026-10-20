############################
# utils_coreset.py
# utility functions for coreset selection case.
############################
import json
from datatools.slice import Slice
import os
import pickle
import numpy as np
import config
from config import DATASET_DIR
from datatools.block_manager import BlockManager

def import_subset(slice:Slice,prefix, subset_path):
    '''
    将原先的子集导入为新的slice
    逐个读取subset_path中的pkl,读取'indices'字段并转换为新的slice
    '''
    for pkl_file in os.listdir(subset_path):
        if pkl_file.endswith('.pkl'):
            with open(os.path.join(subset_path, pkl_file), 'rb') as f:
                data = pickle.load(f)
            subset=slice.save_subset(prefix+pkl_file[:-4], data['indices'])
            print(f"Imported subset {pkl_file} as slice {prefix+pkl_file[:-4]}")
            print(subset.global_ids)


def get_key_words(slice, output_dir,label_to_class):
    '''
    对分类任务用标签生成关键词，所有索引均为valid的
    '''
    valid_indices = slice.get_block_offset()
    labels = slice.get_field('label')
    # 将label通过label_to_class映射为名称
    titles = [label_to_class[label] for label in labels if label < len(label_to_class)]
    print(titles)

    # 保存indices、titles
    indices_path = os.path.join(output_dir, "valid_indices.npy")
    np.save(indices_path, valid_indices)
    titles_path = os.path.join(output_dir, "valid_titles.json")
    with open(titles_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=4)
    print(f"[✓] 提取完成，已保存标题到：{titles_path}")
    key_word_path = os.path.join(output_dir,  "key_words.json")
    with open(key_word_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=4)
    print(f"[✓] 提取完成，已保存关键词到：{key_word_path}")

    bm = BlockManager()
    start_id = slice.get_global_id(0)
    block_info = slice.get_block_info()
    block_info["title"] = {
        "indices_path": os.path.relpath(indices_path, DATASET_DIR),
        "titles_path": os.path.relpath(titles_path, DATASET_DIR),
        "key_word_path": os.path.relpath(key_word_path, DATASET_DIR)
    }
    bm.update_block(start_id, block_info)

def run_utils():
    # 用于test运行各个函数
    # config.MODE = 'coreset'
    s=Slice('CIFAR10-whole')
    # import_subset(s,'CIFAR10-', subsets_path)

    label_to_class = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    get_key_words(s, output_dir=os.path.join(DATASET_DIR, 'CIFAR10'), label_to_class=label_to_class)
