"""Read training checkpoint names without loading model weights."""
import os
import re
from config import RESULT_DIR


def parse_results(results_dir=RESULT_DIR):
    """
    读取 results 目录下的所有 ckpt 文件，返回解析后的结果字典。
    """
    pattern = re.compile(
        r'(?P<dataset>[^_]+)_'               # 数据集
        r'(?P<model>[^_]+)_'                 # 模型
        r'(?P<method>[^_]+)_'                # 方法
        r'(?P<fraction>[^_]+)_'              # 采样率
        r'epoch(?P<epoch>\d+)_'
        r'exp(?P<seed>\d+)_'
        r'(?P<date>[^_]+)_'
        r'(?P<acc>[0-9.]+)\.ckpt'            # 准确率
    )

    results = []

    if not os.path.isdir(results_dir):
        return []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".ckpt"):
            continue
        match = pattern.match(fname)
        if match:
            info = match.groupdict()
            # 转换数据类型
            info["epoch"] = int(info["epoch"])
            info["seed"] = int(info["seed"])
            info["fraction"] = float(info["fraction"])
            info["acc"] = float(info["acc"])
            results.append(info)

    return results
