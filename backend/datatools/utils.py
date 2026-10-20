from functools import lru_cache
import torch
from PIL import Image
import os
import cv2
import numpy as np
import json

from tqdm import tqdm
from datatools.feature_extractor.runner import extract_features_from_slice
from .embedding import load_embedding_model, reduce_features, save_embedding
from config import DATASET_DIR,INDEX_DIR
from datatools.block_manager import BlockManager



def build_feature_and_embedding(
    image_index_path: str,
    block_info: dict,
    output_prefix: str,
    reference_block_info: dict,
    is_small: bool,
    temp_dir: str
):
    """
    根据原 slice 的特征模型和降维器，对 image_index_path 中图像进行特征提取与降维，
    并更新 block_info 字段中的 feature / embedding 路径（以 DATASET_DIR 为相对根）。

    - 如果已有同名特征或 embedding 文件，会跳过计算。
    """

    # ===== 1. 特征提取 =====
    feature_model = reference_block_info.get(
        'feature', {}).get('model', 'resnet50')
    feature_path = output_prefix + '_feature.npy'

    if os.path.exists(feature_path):
        print(f"[✓] 已存在特征文件：{feature_path}，跳过提取")
    else:
        print(f"[→] 开始特征提取（模型: {feature_model}）")
        # extract_features_from_json(
        #     json_path=image_index_path,
        #     output_path=feature_path,
        #     extractor_name=feature_model,
        #     is_small=is_small,
        #     num_chunks=1,
        #     temp_dir=temp_dir
        # )
        print(f"[✓] 特征提取完成：{feature_path}")

    block_info['feature'] = {
        'model': feature_model,
        'path': os.path.relpath(feature_path, start=DATASET_DIR)
    }

    # ===== 2. 降维处理（如有） =====
    # emb_info = reference_block_info.get('embedding')
    # if emb_info:
    #     emb_path = output_prefix + '_embedding.npy'
    #     if os.path.exists(emb_path):
    #         print(f"[✓] 已存在 embedding 文件：{emb_path}，跳过降维")
    #     else:
    #         emb_model_path = emb_info['model']
    #         full_emb_model_path = os.path.join(DATASET_DIR, emb_model_path)
    #         print(f"[→] 加载降维模型：{full_emb_model_path}")
    #         emb_model = load_embedding_model(full_emb_model_path)

    #         features = np.load(feature_path)
    #         embedding = reduce_features(emb_model, features)
    #         save_embedding(embedding, emb_path)
    #         print(f"[✓] 降维完成：{emb_path}")

    #     block_info['embedding'] = {
    #         'model': emb_info['model'],
    #         'path': os.path.relpath(emb_path, start=DATASET_DIR)
    #     }


def create_block_from_json(
    image_json: str,
    label_json: str,
    dataset: str,
    type_: str,
    task: str,
    extractor_name: str = 'resnet50'
):
    """
    主函数：根据图片和标签 json 创建 block，提取特征并保存 slice。
    """
    # 1. 加载图片路径和标签
    with open(image_json, 'r') as f:
        rel_paths = json.load(f)
    # with open(label_json, 'r') as f:
    #     labels = json.load(f)

    # assert len(rel_paths) == len(labels), "图像与标签数量不一致"

    length = len(rel_paths)

    # 2. 构造 block_info（未加 global_id，仅保存原始 image/label 信息）
    image_index_path = os.path.relpath(image_json, DATASET_DIR)
    label_path = os.path.relpath(label_json, DATASET_DIR)
    discription = f"{dataset}-{type_}-{task}"

    block_info = {
        "dataset": dataset,
        "type": type_,
        "task": task,
        "image_index_path": image_index_path,
        "label": {
            "type": "json",
            "path": label_path
        }
        # "feature" 暂未加，待提取完补上
    }

    # 3. 注册 block，获得 start_id
    manager = BlockManager()
    start_id = manager.register_block(
        length=length, discription=discription, block_info=block_info)

    # 4. 构造 global_id 序列
    global_ids = list(range(start_id, start_id + length))

    # 5. 创建 Slice 并保存
    slice_name = f"{dataset}-{type_}"
    from datatools.slice import Slice
    slice_obj = Slice(name=slice_name, global_ids=global_ids, overwrite=True)
    slice_obj.save()

    # 6. 构造 feature 路径（与 image_json 同名加 _feature.npy）
    feature_path = image_json.replace('.json', f'_{extractor_name}.npy')

    # 7. 提取特征并保存
    extract_features_from_slice(
        slice_obj, feature_path, extractor_name=extractor_name)

    # 8. 回写 feature 路径到 block 文件（可选：更新 block json）
    block_file = os.path.join(manager.blocks_dir, f"{start_id:08d}.json")
    with open(block_file, 'r') as f:
        block_data = json.load(f)
    block_data["features"] = {extractor_name:os.path.relpath(feature_path, DATASET_DIR)}
    with open(block_file, 'w') as f:
        json.dump(block_data, f, indent=2)

    print(
        f"[✓] 成功创建 block: {block_file}, slice: {slice_name}, feature: {feature_path}")


def create_tabular_block_from_csv(
    csv_path: str,
    dataset: str = 'Airbnb',
    city: str = 'Austin',
    task: str = 'tabular',
    name_col: str = 'name',
    save_titles: bool = True
):
    """根据一个CSV文件注册一个 tabular block，并创建对应的 slice。

    不影响现有接口：
    - block_info 中新增 `title` 字段，结构与现有 title 兼容：包含 indices_path / titles_path / key_word_path
    - key_word 使用 `name` 列（若不存在则空串）
    - title 使用整行数据（转换为 JSON 字符串，便于后续解析）
    - 生成一个 label 文件（全部填 0）以确保 `label` 字段存在，兼容已有接口

    参数:
    - csv_path: 绝对或相对 (DATASET_DIR) 的CSV路径
    - dataset: 数据集名，用于 block 描述与 slice 命名
    - city: 城市名，用于区分不同 CSV
    - task: 任务类型，默认为 'tabular'
    - name_col: 作为 key_word 的列名
    - save_titles: 是否保存 title JSON（允许后续扩展为节省内存时跳过）
    """
    import pandas as pd
    import numpy as np
    import json
    import os

    # 1. 解析路径并读取数据
    # 支持传入已存在的绝对或相对路径；若传入相对且不存在，则拼 DATASET_DIR
    if os.path.exists(csv_path):
        abs_csv_path = os.path.abspath(csv_path)
    else:
        abs_csv_path = os.path.abspath(os.path.join(DATASET_DIR, csv_path))
    if not os.path.exists(abs_csv_path):
        raise FileNotFoundError(f"CSV 不存在: {abs_csv_path}")
    df = pd.read_csv(abs_csv_path)
    length = len(df)
    if length == 0:
        raise ValueError("CSV 内容为空，无法注册 block")

    # 2. 构造标题与关键词
    # key_word: 使用 name 列
    if name_col in df.columns:
        key_words = df[name_col].fillna('').astype(str).tolist()
    else:
        key_words = [''] * length

    # title: 整行数据 -> JSON 字符串（保持原始列顺序）
    if save_titles:
        titles = [
            json.dumps(row._asdict() if hasattr(row, '_asdict') else row.to_dict(), ensure_ascii=False)
            for _, row in df.iterrows()
        ]
    else:
        titles = [''] * length

    # 3. 准备输出文件（相对 DATASET_DIR）
    base_name = f"{dataset}_{city}"  # 避免与其他类型冲突
    rel_csv_path = os.path.relpath(abs_csv_path, os.path.abspath(DATASET_DIR))
    indices_rel = f"{base_name}_title_indices.npy"
    titles_rel = f"{base_name}_titles.json"
    key_words_rel = f"{base_name}_keywords.json"
    labels_rel = f"{base_name}_labels.json"

    np.save(os.path.join(DATASET_DIR, indices_rel), np.arange(length, dtype=np.int64))
    with open(os.path.join(DATASET_DIR, key_words_rel), 'w', encoding='utf-8') as f:
        json.dump(key_words, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATASET_DIR, titles_rel), 'w', encoding='utf-8') as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATASET_DIR, labels_rel), 'w', encoding='utf-8') as f:
        json.dump([0]*length, f, ensure_ascii=False)

    # 4. 构造 block_info —— 不使用 image_index_path，保留 label 与 title 结构兼容
    discription = f"{dataset}-{city}-{task}"  # 与已有搜索逻辑保持相似格式
    block_info = {
        "dataset": dataset,
        "type": city,
        "task": task,
        "csv_path": rel_csv_path,
        "label": {
            "type": "json",
            "path": labels_rel
        },
        "title": {
            "indices_path": indices_rel,
            "titles_path": titles_rel,
            "key_word_path": key_words_rel
        }
    }

    # 5. 注册 block
    manager = BlockManager()
    start_id = manager.register_block(length=length, discription=discription, block_info=block_info)

    # 6. 创建全局ID与 slice
    global_ids = list(range(start_id, start_id + length))
    from datatools.slice import Slice
    slice_name = f"{dataset}-{city}"  # 保持与 create_block_from_json 风格一致
    slice_obj = Slice(name=slice_name, global_ids=global_ids, overwrite=True)
    slice_obj.save()

    print(f"[✓] Tabular block 注册成功: {discription} | 行数: {length} | slice: {slice_name}")
    return slice_obj


def create_imdb_block_from_csv(
    csv_path: str,
    dataset: str = 'IMDB',
    task: str = 'text',
    review_col: str = 'review',
    sentiment_col: str = 'sentiment',
    movie_col: str = 'movie',
    positive_values = ('positive','pos','1'),
    negative_values = ('negative','neg','0')
):
    """注册 IMDB 文本评论数据集为一个 block，并根据情感生成正/负两个 slice。

    要求 CSV 至少包含评论列 (review_col) 和情感列 (sentiment_col)。
    关键字 key_word 来自 movie_col 若存在，否则从评论中用正则提取首个连续的首字母大写片段。
    标题 title 即整条评论文本。

    生成内容：
    - indices.npy ：行索引
    - titles.json ：评论文本列表
    - keywords.json ：关键字（电影名或提取片段）
    - sentiments.json ：情感 0/1 标签
    - features_IMDB.npy ：延迟由外部脚本生成（此函数只做注册）
    - 两个 slice: IMDB-positive / IMDB-negative
    """
    import pandas as pd, os, re, json, numpy as np

    # 解析路径
    if os.path.exists(csv_path):
        abs_csv = os.path.abspath(csv_path)
    else:
        abs_csv = os.path.abspath(os.path.join(DATASET_DIR, csv_path))
    if not os.path.exists(abs_csv):
        raise FileNotFoundError(f"IMDB CSV 不存在: {abs_csv}")

    df = pd.read_csv(abs_csv)
    if review_col not in df.columns:
        raise ValueError(f"CSV 缺少评论列: {review_col}")
    if sentiment_col not in df.columns:
        raise ValueError(f"CSV 缺少情感列: {sentiment_col}")

    n = len(df)
    if n == 0:
        raise ValueError("IMDB CSV 为空")

    reviews = df[review_col].fillna('').astype(str).tolist()

    # 关键字：优先 movie_col，其次 regex 提取
    if movie_col in df.columns:
        raw_movies = df[movie_col].fillna('').astype(str).tolist()
    else:
        raw_movies = [''] * n

    def extract_first_cap_phrase(text: str) -> str:
        # 匹配连续的首字母大写单词序列（最多 6 个），更宽松
        pattern = re.compile(r'(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,5})')
        m = pattern.search(text)
        return m.group(0) if m else ''

    def extract_quoted_title(text: str, min_len: int = 10) -> str:
        # 提取引号中的标题（中文/英文引号）
        for rx in [r'"([^\"]{'+str(min_len)+r',})"', r'“([^”]{'+str(min_len)+r',})”', r'‘([^’]{'+str(min_len)+r',})’']:
            m = re.search(rx, text)
            if m:
                return m.group(1).strip()
        return ''

    def improve_keyword(review: str, current: str) -> str:
        # 已有且够长，直接返回
        if isinstance(current, str) and len(current.strip()) >= 10:
            return current.strip()
        # 尝试引号标题
        cand = extract_quoted_title(review)
        if len(cand) >= 10:
            return cand
        # 尝试首字母大写短语
        cap = extract_first_cap_phrase(review)
        if len(cap) >= 10:
            return cap
        # 回退：取评论中最长的单词/短语（限制长度避免过长）
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", review)
        tokens.sort(key=len, reverse=True)
        for t in tokens[:5]:
            if len(t) >= 10:
                return t[:60]
        return 'UNKNOWN'

    key_words = [improve_keyword(r, m) for m, r in zip(raw_movies, reviews)]

    # 情感：映射到 0/1
    sentiments_raw = df[sentiment_col].astype(str).str.lower().tolist()
    sentiment_labels = []
    for s in sentiments_raw:
        if s in [v.lower() for v in positive_values]:
            sentiment_labels.append(1)
        elif s in [v.lower() for v in negative_values]:
            sentiment_labels.append(0)
        else:
            # 未知情感，默认 0
            sentiment_labels.append(0)

    # 输出文件名前缀
    rel_csv = os.path.relpath(abs_csv, os.path.abspath(DATASET_DIR))
    base = 'IMDB'
    indices_rel = f'{base}_title_indices.npy'
    titles_rel = f'{base}_titles.json'
    keywords_rel = f'{base}_keywords.json'
    sentiments_rel = f'{base}_sentiments.json'

    np.save(os.path.join(DATASET_DIR, indices_rel), np.arange(n, dtype=np.int64))
    with open(os.path.join(DATASET_DIR, titles_rel), 'w', encoding='utf-8') as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATASET_DIR, keywords_rel), 'w', encoding='utf-8') as f:
        json.dump(key_words, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATASET_DIR, sentiments_rel), 'w', encoding='utf-8') as f:
        json.dump(sentiment_labels, f, ensure_ascii=False)

    # block_info
    discription = f'{dataset}-reviews-{task}'
    block_info = {
        'dataset': dataset,
        'type': 'reviews',
        'task': task,
        'csv_path': rel_csv,
        'label': {
            'type': 'json',
            'path': sentiments_rel
        },
        'title': {
            'indices_path': indices_rel,
            'titles_path': titles_rel,
            'key_word_path': keywords_rel
        }
    }

    manager = BlockManager()
    start_id = manager.register_block(length=n, discription=discription, block_info=block_info)
    global_ids = list(range(start_id, start_id + n))

    from datatools.slice import Slice
    # 生成整体 slice 及正/负子 slice
    all_slice = Slice(name=f'{dataset}-all', global_ids=global_ids, overwrite=True)
    all_slice.save()
    pos_ids = [global_ids[i] for i, lab in enumerate(sentiment_labels) if lab == 1]
    neg_ids = [global_ids[i] for i, lab in enumerate(sentiment_labels) if lab == 0]
    Slice(name=f'{dataset}-positive', global_ids=pos_ids, overwrite=True).save()
    Slice(name=f'{dataset}-negative', global_ids=neg_ids, overwrite=True).save()

    print(f"[✓] IMDB block 注册成功: {discription} | 样本: {n} | 正样本: {len(pos_ids)} | 负样本: {len(neg_ids)}")
    return {
        'start_id': start_id,
        'total': n,
        'positive': len(pos_ids),
        'negative': len(neg_ids)
    }


def generate_image_index(dataset_root: str, search_root: str, index_name: str = "index"):
    """
    在 search_root 中递归查找所有图片，生成相对于 dataset_root 的路径列表，并保存为 index.json 到 dataset_root/。

    Args:
        dataset_root (str): 相对路径的参考起点目录。
        search_root (str): 实际查找图片的目录。
        index_json_dirname (str): 保存 index.json 的子目录名称，默认 "index"。
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_paths = []

    for root, dirs, files in os.walk(search_root):
        dirs.sort()
        files.sort()

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                abs_path = os.path.join(root, file)
                try:
                    rel_path = os.path.relpath(abs_path, dataset_root)
                except ValueError as e:
                    raise ValueError(
                        f"图片路径 {abs_path} 不在 dataset_root {dataset_root} 下，无法生成相对路径") from e
                image_paths.append(rel_path)

    # 构建输出路径
    output_path = os.path.join(dataset_root, index_name+".json")

    # 保存为 JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, indent=2, ensure_ascii=False)

    print(f"索引文件已生成，共 {len(image_paths)} 张图片，保存至：{output_path}")



@lru_cache(maxsize=1)
def _get_chart_clip():
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    chart_prompts = ["a chart"]
    text_tokens = clip.tokenize(chart_prompts).to(device)
    with torch.no_grad():
        cached_text_features = clip_model.encode_text(text_tokens)
        cached_text_features /= cached_text_features.norm(dim=-1, keepdim=True)
    return device, clip_model, preprocess, chart_prompts, cached_text_features


def generate_chart_thumbnail(image: np.ndarray, output_size: int = 224) -> np.ndarray:
    """
    从信息图中裁剪出图表主体，并以裁剪方式生成一个正方形缩略图，尽量避免填充。
    """
    ori_image= image.copy()
    # --- 0. 预处理 ---
    #image=suppress_text_regions(image)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 1. 自适应阈值反显图表区域 ---
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 55, 10)

    # --- 2. 闭操作连接结构 ---
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))

    # --- 3. 去除小连通域 ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph)
    min_area = 0.02 * h * w
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            x, y, bw, bh = stats[i, :4]
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

    # --- 4. 找最大区域 ---
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (output_size, output_size))
    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # --- 5. 从中裁剪最大正方形区域 ---
    cx = x + bw // 2
    cy = y + bh // 2
    box_size = min(bw, bh)

    # 根据中心点向四周取正方形区域，尽可能大，不能越界
    half = box_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    # 防止小图像截出来不是正方形
    crop = ori_image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side = min(ch, cw)
    crop = crop[0:side, 0:side]

    # --- 6. 缩放到标准缩略图大小 ---
    thumbnail = cv2.resize(crop, (output_size, output_size),
                           interpolation=cv2.INTER_AREA)
    return thumbnail


def crop_chart_region_precise(image: np.ndarray) -> np.ndarray:
    """
    高质量裁剪图表区域：
    - 剔除背景；
    - 利用边缘 + 形态学操作；
    - 提取最可能的主体区域；
    - 输出正方形裁剪。
    """
    ori_image = image.copy()
    image = suppress_text_regions(image)
    h, w = image.shape[:2]

    # 1. 灰度 + 去背景（自适应处理）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))  # 检测白背景
    bg_mask = cv2.dilate(bg_mask, np.ones((3, 3), np.uint8), iterations=1)
    fg_mask = cv2.bitwise_not(bg_mask)

    # 2. 边缘检测增强图表结构
    edges = cv2.Canny(gray, 80, 200)
    edges = cv2.bitwise_and(edges, fg_mask)

    # 3. 连通域分析
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        ar = cw / ch if ch > 0 else 0
        if area > h * w * 0.02 and 0.2 < ar < 6:
            boxes.append((x, y, cw, ch))

    if not boxes:
        return None

    # 4. 合并区域（取最大外接）
    x0 = min([x for x, y, cw, ch in boxes])
    y0 = min([y for x, y, cw, ch in boxes])
    x1 = max([x+cw for x, y, cw, ch in boxes])
    y1 = max([y+ch for x, y, cw, ch in boxes])

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0)
    half = side // 2

    # 保证区域不出界,转为正方形
    if w < h:
        half = min(half, w // 2)
    else:
        half = min(half, h // 2)
    left = max(cx - half, 0)
    top = max(cy - half, 0)
    right = min(cx + half, w)
    bottom = min(cy + half, h)

    cropped = ori_image[top:bottom, left:right]
    return cropped



def crop_chart_projection(image: np.ndarray) -> np.ndarray:
    """
    利用图像投影法裁剪柱状图等结构性强的图表区域。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)

    # 阈值处理突出前景（柱状图）
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去除边缘小噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 1. 水平投影（Y方向）：计算每一行的前景像素数
    horizontal_proj = np.sum(binary, axis=1)
    top, bottom = _find_main_bounds_from_projection(
        horizontal_proj, min_range=h//5)

    # 2. 垂直投影（X方向）：计算每一列的前景像素数
    vertical_proj = np.sum(binary, axis=0)
    left, right = _find_main_bounds_from_projection(
        vertical_proj, min_range=w//5)

    if top >= bottom or left >= right:
        return None

    # 转为正方形
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    side = max(right - left, bottom - top)
    half = side // 2

    x0 = max(cx - half, 0)
    y0 = max(cy - half, 0)
    x1 = min(cx + half, w)
    y1 = min(cy + half, h)

    return image[y0:y1, x0:x1]


def _find_main_bounds_from_projection(proj: np.ndarray, min_range=50, threshold_ratio=0.15):
    """
    从投影曲线中提取主区域（前景密集区域）
    """
    max_val = np.max(proj)
    threshold = max_val * threshold_ratio

    indices = np.where(proj > threshold)[0]
    if len(indices) == 0:
        return 0, len(proj)

    start = indices[0]
    end = indices[-1]

    if end - start < min_range:
        return 0, len(proj)

    return start, end


def suppress_text_regions(image: np.ndarray) -> np.ndarray:
    """
    尝试抑制图像中的文字区域（不依赖OCR）：
    - 基于轮廓特征；
    - 用背景色填充；
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 二值化 + 文字一般是黑字白底
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 小膨胀连接字符
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_out = image.copy()
    h, w = image.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / ch if ch > 0 else 0

        if area < w * h * 0.02 and 1.5 < aspect < 10 and ch < h * 0.2:
            # 填充为背景色（可选使用邻近区域颜色）
            cv2.rectangle(img_out, (x, y), (x+cw, y+ch), (255, 255, 255), -1)

    return img_out


def smart_crop_chart(image: np.ndarray):
    """
    多策略图表裁剪器：尝试不同裁剪方式，提高成功率，裁剪结束回归原图
    """
    for method in [generate_chart_thumbnail, crop_chart_region_precise, crop_chart_projection]:
        result = method(image)
        if result is not None:
            return result
    return None


def is_chart_like(image: np.ndarray, threshold: float = 0.19,print_detail=False):
    """
    使用 CLIP 判断图像是否像信息图表,打印每个类别的相似度。
    """
    device, clip_model, preprocess, chart_prompts, cached_text_features = _get_chart_clip()
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_img).unsqueeze(0).to(device)

    info=[]
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if print_detail:
            for i in range(cached_text_features.shape[0]):
                similarity = (image_features @
                            cached_text_features[i].unsqueeze(0).T).item()
                info.append(
                    f"类别 {i} ({chart_prompts[i]}): 相似度 = {similarity:.4f}")
                print(f"类别 {i} ({chart_prompts[i]}): 相似度 = {similarity:.4f}")

        else:
            for i in range(cached_text_features.shape[0]):
                similarity = (image_features @
                            cached_text_features[i].unsqueeze(0).T).item()
                info.append(f"类别 {i} ({chart_prompts[i]}): 相似度 = {similarity:.4f}")
                if similarity >= threshold:
                    return True,info


    return False,info


def generate_thumbnails_with_clip(slice, output_dir: str,name:str, threshold: float = 0.22):
    """
    生成图表缩略图，使用CLIP判断是否保留,如果已存在缩略图则跳过：
    - 每张图裁剪主区域；
    - 判断是否图表样式；
    - 按每1w个分组保存；
    - index.json 保存所有相对路径或空串。
    """
    thumb_dir = os.path.join(output_dir, 'thumbnails_'+name)
    os.makedirs(thumb_dir, exist_ok=True)

    index = []
    infos={}

    for i in tqdm(range(len(slice)), desc=f"生成缩略图 [{slice.name}]"):
        # if i>100:
        #     break
        gid = slice.get_global_id(i)
        image_path = slice.get_image_path(i)
        # 子文件夹划分（每10000个）
        group_id = gid // 10000 * 10000
        group_dir = os.path.join(thumb_dir, f"part_{group_id:05d}")
        os.makedirs(group_dir, exist_ok=True)

        thumb_name = f"sample-{gid}_thumb.jpg"
        thumb_path = os.path.join(group_dir, thumb_name)
        # if os.path.exists(thumb_path):
        #     index.append(os.path.relpath(thumb_path, output_dir))
        #     continue
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("图像读取失败")

            cropped = smart_crop_chart(image)
            if cropped is None:
                index.append("")
                infos[gid]=slice.get_image_path(i) + " - 裁剪失败"
                continue

            cv2.imwrite(thumb_path, cropped)
            like, info = is_chart_like(cropped, threshold=threshold)
            infos[gid] = info
            if not like:
                index.append("")  # 不像图表则跳过
                continue
            rel_path = os.path.relpath(thumb_path, output_dir)
            index.append(rel_path)

        except Exception as e:
            print(f"[×] 处理失败：{image_path} | 错误：{e}")
            index.append("")

    index_path = os.path.join(output_dir, 'thumbnails_'+name+'.json')
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    # 保存 infos
    infos_path = os.path.join(output_dir, 'thumbnails_'+name+'_infos.json')
    with open(infos_path, "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=2, ensure_ascii=False)
    print(f"[✓] 缩略图索引已保存：{infos_path}")
    print(f"[✓] 缩略图已生成，索引文件保存至：{index_path}")
