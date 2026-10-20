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

def generate_subset(slice: Slice, name: str, number: int, class_ratios):
    '''
    为CIFAR10生成子集，并保存为slice
    通过number指定总数目，列表指定每种类别之间的数量比例，每个值为0-1之间。
    '''
    labels = slice.get_field('label')
    class_counts = [int(ratio * number) for ratio in class_ratios]
    print(f"Class counts for subset: {class_counts}")
    print(f"Total samples requested: {number}")

    subset_indices = []
    for class_idx, count in enumerate(class_counts):
        indices = np.where(np.array(labels) == class_idx)[0]
        if len(indices) < count:
            print(f"Warning: Not enough samples for class {class_idx}, using {len(indices)} instead of {count}")
            selected = indices
        else:
            selected = np.random.choice(indices, count, replace=False)
        subset_indices.extend(selected.tolist())

    return slice.save_subset(name, subset_indices)

def extend_subset(slice: Slice, ori_slice: Slice, name: str, additional_indices: list):
    '''
    扩展已有的子集
    '''
    return slice.save_subset(name, ori_slice.get_block_offset().extend(additional_indices))

def run_utils():
    # 用于test运行各个函数
    s = Slice('CIFAR10-whole')

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
    # ratio_blue_dict = {
    #     "airplane": 0,
    #     "automobile": 1.2,
    #     "bird": 1,
    #     "cat": 1,
    #     "deer": 1.2,
    #     "dog": 1,
    #     "frog": 1.3,
    #     "horse": 1,
    #     "ship": 1,
    #     "truck": 1.4
    # }
    # ratio_red_dict = {
    #     "airplane": 1,
    #     "automobile": 0.8,
    #     "bird": 1.5,
    #     "cat": 1,
    #     "deer": 1,
    #     "dog": 1,
    #     "frog": 1,
    #     "horse": 1,
    #     "ship": 1.3,
    #     "truck": 1
    # }
    ratio_blue_dict = {
        "airplane": 1,
        "automobile": 1,
        "bird": 1,
        "cat": 1,
        "deer": 1,
        "dog": 1,
        "frog": 1.3,
        "horse": 1.5,
        "ship": 0,
        "truck": 1.3
    }
    ratio_red_dict = {
        "airplane": 0.5,
        "automobile": 0.7,
        "bird": 1.2,
        "cat": 1.5,
        "deer": 1,
        "dog": 1.5,
        "frog": 1,
        "horse": 1,
        "ship": 1,
        "truck": 1
    }
    ratio_blue = [ratio_blue_dict[cls] for cls in label_to_class]
    ratio_red = [ratio_red_dict[cls] for cls in label_to_class]

    sub = generate_subset(s,
                          name="CIFAR10-B-blue",
                          number=1000,
                          class_ratios=ratio_blue
                          )
    sub = generate_subset(s,
                          name="CIFAR10-B-red",
                          number=1000,
                          class_ratios=ratio_red
                          )
    print(f"Subset created: {sub.name} with {len(sub)} images.")
