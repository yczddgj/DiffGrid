import shutil
from modelscope import AutoProcessor
from torch import nn
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import clip
import torch
from PIL import Image
import os
import cv2
import numpy as np
import json

from tqdm import tqdm
from datatools.reducer.runner import get_slice_different_embedding
from datatools.block_manager import BlockManager
from datatools.slice import Slice
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor
from config import DATASET_DIR, MODEL_DIR


def poisson_disk_sampling(embedding: np.ndarray, all_indices: np.ndarray, radius: float = 0.1) -> np.ndarray:
    """
    对要显示图片的点进行泊松圆盘采样

    Args:
        embedding: 嵌入坐标
        all_indices: 所有候选点的索引
        radius: 采样半径
    """
    if len(all_indices) == 0:
        return []

    shuffled_indices = np.random.permutation(all_indices)
    sampled_indices = []

    emb = embedding[shuffled_indices]  # shape: (N, D)
    selected_mask = np.zeros(len(shuffled_indices), dtype=bool)

    for i in range(len(shuffled_indices)):
        if selected_mask[i]:
            continue
        current = emb[i]
        sampled_indices.append(shuffled_indices[i])
        # 计算当前点与后续所有点的距离
        if i + 1 < len(shuffled_indices):
            rest = emb[i + 1:]
            dists = np.linalg.norm(rest - current, axis=1)
            selected_mask[i + 1:] |= dists < radius

    return np.array(sampled_indices)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx]


def get_labels_for_thumbnails(slice, contrast_slice, output_dir: str, name: str,
                              model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                              device: str = "cuda" if torch.cuda.is_available() else "cpu",
                              batch_size: int = 8):
    """
    对于有缩略图的样本，根据embedding进行采样，使用 Qwen2.5-VL 生成图表标题标签。
    多卡并行推理，防止设备不一致问题，加入双候选+判断机制提升结果质量。
    """

    # ========== 1. 获取采样索引 ==========
    thumb_indices = slice.get_thumb_indices()
    embedding = get_slice_different_embedding(slice, slice, contrast_slice)
    indices = poisson_disk_sampling(embedding, thumb_indices, radius=0.1)
    indices = np.sort(np.array(indices, dtype=int))
    print(f"[✓] 采样缩略图索引：{len(indices)}")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, name + "_labeled_indices.npy"), indices)

    # ========== 2. 加载模型（DataParallel防多卡冲突） ==========
    print(f"[✓] 加载模型：{model_name}")
    local_model_dir = str(MODEL_DIR / 'qwen2.5')
    processor = AutoProcessor.from_pretrained(local_model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model_dir, torch_dtype="auto")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()

    # ========== 3. 设置提示词 ==========
    prompt_text = "Please extract the main title of this chart. Output only the title text in English, without any explanation."
    judge_text_template = (
        "Given the two texts below, please determine which one is more like a main title of an English chart. Output only '1' or '2'.\n"
        "1. {}\n2. {}"
    )

    # ========== 4. 构造 DataLoader ==========
    image_paths = slice.get_image_paths(indices)
    dataset = ImagePathDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    labels = []

    def process_batch(batch_image_paths):
        batch_labels = []
        for image_path in batch_image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                # -------- Step 1: 提示构造 --------
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]}
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(device)

                # -------- Step 2: 生成两个候选 --------
                with torch.no_grad():
                    outputs = model.module.generate(  # DataParallel
                        **inputs,
                        max_new_tokens=64,
                        num_beams=5,
                        num_return_sequences=2,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.1,
                    )
                candidates = processor.batch_decode(
                    outputs, skip_special_tokens=True)

                # -------- Step 3: 生成选择提示 --------
                judge_prompt = judge_text_template.format(
                    candidates[0], candidates[1])
                judge_msg = [{"role": "user", "content": judge_prompt}]
                judge_text = processor.apply_chat_template(
                    judge_msg, tokenize=False, add_generation_prompt=True)
                judge_inputs = processor(
                    text=[judge_text], padding=True, return_tensors="pt").to(device)

                with torch.no_grad():
                    judge_output = model.module.generate(
                        **judge_inputs, max_new_tokens=1)
                choice = processor.batch_decode(
                    judge_output, skip_special_tokens=True)[0].strip().split("\n")[-1].strip()

                if choice == "1":
                    final = candidates[0].strip()
                elif choice == "2":
                    final = candidates[1].strip()
                else:
                    final = candidates[0].strip()  # fallback

                print(f"[✓] 图 {image_path} → {final}")
                batch_labels.append(final)

            except Exception as e:
                print(f"[×] 处理失败 {image_path}：{e}")
                batch_labels.append("")

        return batch_labels

    # ========== 5. 多线程批量处理 ==========
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_batch, batch)
                   for batch in dataloader]
        for future in tqdm(futures, desc="生成标签"):
            labels.extend(future.result())

    # ========== 6. 保存输出 ==========
    output_path = os.path.join(output_dir, name + "_title_labels.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"[✓] 标签保存完毕，共生成：{len(labels)} 条")


def check_labels_only(slice: Slice, output_dir, name):
    '''
    检查图片路径和标签，并处理为最终可用标题
    '''
    indices = np.load(os.path.join(output_dir, name + "_labeled_indices.npy"))
    labels = json.load(open(os.path.join(output_dir, name +
                       "_title_labels.json"), "r", encoding="utf-8"))

    # 1. 筛选可用的标签和对应索引
    valid_labels = []
    valid_indices = []
    for i, label in enumerate(labels):
        if isinstance(label, str) and label != "" and label.strip():
            # 处理标签，去除前缀和后缀的换行符
            label = label.split("\n")[-1].strip()
            valid_labels.append(label)
            valid_indices.append(indices[i])

    # 2. 利用模型检查缩略图的质量
    image_paths = slice.get_image_paths(valid_indices)

    # 3. 保存有效标签和索引
    valid_output = {
        "valid_labels": valid_labels,
        "image_paths": image_paths
    }
    # print(valid_indices)
    valid_output_path = os.path.join(output_dir, name + "_valid_labels.json")
    with open(valid_output_path, "w", encoding="utf-8") as f:
        json.dump(valid_output, f, ensure_ascii=False, indent=4)
    # 保存索引npy和title字符串json
    np.save(os.path.join(output_dir, name + "_valid_indices.npy"),
            np.array(valid_indices))
    with open(os.path.join(output_dir, name + "_valid_titles.json"), "w", encoding="utf-8") as f:
        json.dump(valid_labels, f, ensure_ascii=False, indent=4)

    print(f"[✓] 有效标签数量：{len(valid_labels)}, 有效索引数量：{len(valid_indices)}")

    bm = BlockManager()
    start_id = slice.get_global_id(0)
    block_info = slice.get_block_info()
    block_info["title"] = {
        "indices_path": os.path.relpath(os.path.join(output_dir, name + "_valid_indices.npy"), DATASET_DIR),
        "titles_path": os.path.relpath(os.path.join(output_dir, name + "_valid_titles.json"), DATASET_DIR)
    }
    bm.update_block(start_id, block_info)


def parse_and_validate_box(box_text, image_width, image_height):
    """
    解析并验证边界框格式和合理性，格式为: "x, y, width, height"
    """
    try:
        box = [int(float(x.strip())) for x in box_text.strip().split(",")]
        if len(box) != 4:
            raise ValueError("Box should have exactly 4 values.")
        x, y, w, h = box

        # 合理性判断
        if min(x, y, w, h) < 0:
            raise ValueError("Negative value in box.")
        if w != h:
            raise ValueError("Width and height must be equal (square box).")
        if x + w > image_width or y + h > image_height:
            raise ValueError("Box exceeds image boundaries.")

        return box

    except Exception as e:
        print(f"[×] 无效边界框格式：'{box_text}'，原因：{e}")
        return None


def crop_image_with_box(image_path, box):
    """
    根据左上角和宽度裁剪图像为正方形
    """
    image = cv2.imread(image_path)
    x, y, w, h = box
    x, y, w, h = map(int, [x, y, w, h])
    cropped = image[y:y+h, x:x+w]
    return cropped


def check_thumbnails(slice, output_dir, name,
                     model_path=str(MODEL_DIR / 'qwen2.5'),
                     device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    检查缩略图质量。对质量不佳的，通过 Qwen 预测边界框并裁剪，
    把所有新裁剪图统一存入 cache/{name} 目录，等待用户确认。
    """
    # —— 1. 读取有效标签 & 索引 ——
    valid_indices = np.load(os.path.join(
        output_dir, name + "_valid_indices.npy"))
    valid_labels = json.load(open(os.path.join(output_dir, name +
                                               "_valid_titles.json"), "r", encoding="utf-8"))

    # —— 2. 载入模型 ——
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()

    judge_prompt = (
        "Does this thumbnail show the main visual content of a chart, such as bars, lines, or pies, "
        "rather than just titles, or text? Answer only 'yes' or 'no'."
    )
    box_prompt = (
        "Provide a square bounding box that best captures the main visual elements of the chart, "
        "such as bars, lines, pies or data regions. Avoid selecting areas that mainly contain titles, "
        "decorative text, or logos.\n\n"
        "Return the bounding box in the exact format: x, y, width, height — where:\n"
        "- (x, y) is the top-left corner of the box,\n"
        "- width = height,\n"
        "- All values are integers,\n"
        "Do not include any extra explanation. Only output the bounding box. Example: '120, 80, 300, 300'"
    )

    thumb_paths = slice.get_thumbnail_paths(valid_indices)
    image_paths = slice.get_image_paths(valid_indices)

    # 准备 cache 目录
    cache_dir = os.path.join(output_dir, "cache", name)
    os.makedirs(cache_dir, exist_ok=True)

    # —— 3. 遍历判断 & 裁剪到 cache ——
    for thumb_path, img_path in tqdm(zip(thumb_paths, image_paths), total=len(thumb_paths), desc="check & cache"):
        try:
            # 3.1 判断原 thumbnail 是否合格
            img = Image.open(thumb_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img}, {"type": "text", "text": judge_prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            ims, vids = process_vision_info(messages)
            inputs = processor(
                text=[text], images=ims, videos=vids, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = (model.module.generate(**inputs, max_new_tokens=1)
                       if isinstance(model, nn.DataParallel)
                       else model.generate(**inputs, max_new_tokens=1))
            ans = processor.batch_decode(out, skip_special_tokens=True)[
                0].lower().split("\n")[-1].strip()
            if ans.startswith("yes"):

                continue  # 好的直接跳过

            # 3.2 生成边界框并裁剪
            orig = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": orig}, {"type": "text", "text": box_prompt}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            ims, vids = process_vision_info(messages)
            inputs = processor(
                text=[text], images=ims, videos=vids, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = (model.module.generate(**inputs, max_new_tokens=32)
                       if isinstance(model, nn.DataParallel)
                       else model.generate(**inputs, max_new_tokens=32))
            box_txt = processor.batch_decode(out, skip_special_tokens=True)[
                0].split("\n")[-1].strip()
            box = parse_and_validate_box(box_txt, orig.width, orig.height)
            if box is None:
                continue

            # 裁剪并写入 cache
            x, y, w, h = box
            orig_cv = cv2.imread(img_path)
            crop = orig_cv[y:y+h, x:x+w]
            # 统一输出为 thumb 名字到 cache
            base = os.path.basename(thumb_path)
            thumb_name, ext = os.path.splitext(base)

            # 保存裁剪图到 cache
            crop_path = os.path.join(cache_dir, base)
            cv2.imwrite(crop_path, crop)

            # 复制当前缩略图到 cache（带 _origin 后缀）
            origin_thumb = cv2.imread(thumb_path)
            origin_path = os.path.join(cache_dir, f"{thumb_name}_origin{ext}")
            cv2.imwrite(origin_path, origin_thumb)

        except Exception as e:
            print(f"[×] 处理失败 {thumb_path}: {e}")

    print(f"[✓] 裁剪后的候选已保存到：{cache_dir}，请 review 后再执行 apply_cached_thumbnails()")


def apply_cached_thumbnails(slice, output_dir, name):
    """
    将 cache/{name} 中经人工确认的缩略图替换到正式位置：
    - 原 thumb 重命名为 _bad
    - cache 中的文件复制到 thumb
    - 同时备份当前 thumb 为 _qwen
    """
    # 加载索引
    valid_indices = np.load(os.path.join(
        output_dir, name + "_valid_indices.npy"))
    thumb_paths = slice.get_thumbnail_paths(valid_indices.astype(int))

    cache_dir = os.path.join(output_dir, "cache", name)

    for thumb_path in tqdm(thumb_paths, desc="apply cache"):
        base = os.path.basename(thumb_path)
        cached = os.path.join(cache_dir, base)
        if not os.path.exists(cached):
            continue  # cache 中无更新，跳过

        root, ext = os.path.splitext(thumb_path)
        bad_p = root + "_bad" + ext
        qwen_p = root + "_qwen" + ext

        # 1. 备份现 thumb 到 _qwen
        if os.path.exists(thumb_path):
            shutil.move(thumb_path, qwen_p)
        # 2. 恢复原 thumb（从 cached）
        shutil.copy(cached, thumb_path)
        # 3. 把原始 _bad（若存在）改名回正式 thumb
        if os.path.exists(bad_p):
            shutil.move(bad_p, thumb_path)

    print(f"[✓] 已应用 cache 中的缩略图替换，正式目录：{output_dir}")


def recover_labels(slice, output_dir, name):
    '''
    恢复缩略图：将被替换的缩略图恢复为原始版本（_bad），
    同时将 Qwen 替换过的缩略图备份为 _qwen。
    '''
    indices = np.load(os.path.join(output_dir, name + "_labeled_indices.npy"))
    labels = json.load(open(os.path.join(output_dir, name +
                       "_title_labels.json"), "r", encoding="utf-8"))

    # Step 1: 筛选有效标签
    valid_labels, valid_indices = [], []
    for i, label in enumerate(labels):
        if isinstance(label, str) and label.strip():
            label = label.split("\n")[-1].strip()
            valid_labels.append(label)
            valid_indices.append(indices[i])

    print(f"[✓] 开始恢复缩略图，共 {len(valid_indices)} 张")

    thumb_paths = slice.get_thumbnail_paths(valid_indices)
    count = 0
    for thumb_path in tqdm(thumb_paths, desc="恢复缩略图"):
        root, ext = os.path.splitext(thumb_path)
        bad_path = root + "_bad" + ext
        qwen_path = root + "_qwen" + ext

        if os.path.exists(bad_path):
            try:
                # 当前缩略图重命名为 _qwen 备份
                if os.path.exists(thumb_path):
                    shutil.move(thumb_path, qwen_path)
                # 恢复原图
                shutil.move(bad_path, thumb_path)
                print(f"[✓] 恢复成功：{os.path.basename(thumb_path)}")
                count += 1
            except Exception as e:
                print(f"[×] 恢复失败：{thumb_path}，原因：{e}")
        else:
            print(f"[ ] 无需恢复：{os.path.basename(thumb_path)}")

    print(f"[✓] 缩略图恢复完成，共处理：{count} 张")


def get_key_words(slice, output_dir, name,
                  model_path=str(MODEL_DIR / 'qwen2.5'),
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.to(device)
    model.eval()

    valid_indices = np.load(os.path.join(
        output_dir, name + "_valid_indices.npy"))
    valid_labels = json.load(open(os.path.join(
        output_dir, name + "_valid_titles.json"), "r", encoding="utf-8"))
    image_paths = slice.get_image_paths(valid_indices)

    keyword_prompt = (
        "Please extract a concise English keyword that best describe the topic of this chart. "
        "Use as few words as possible, avoid numbers and explanations. Output only the keyword."
    )

    key_words = []

    for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Extracting keywords"):
        try:
            title = valid_labels[idx]
            orig = Image.open(img_path).convert("RGB")

            keyword_prompt = (
                f"Title: \"{title}\"\n"
                "Based on the image and this title, extract a concise English keyword that best describe the topic of this chart. "
                "Use as few words as possible, avoid numbers and explanations. Output only the keyword."
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": orig},
                {"type": "text", "text": keyword_prompt}
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            ims, vids = process_vision_info(messages)
            inputs = processor(
                text=[text], images=ims, videos=vids, padding=True, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out = (model.module.generate(
                    **inputs,
                    max_new_tokens=3,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1
                ) if isinstance(model, nn.DataParallel)
                    else model.generate(
                    **inputs,
                    max_new_tokens=3,
                    num_beams=3,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1
                ))

            candidate = processor.batch_decode(out, skip_special_tokens=True)[0]
            print(candidate)
            candidate = candidate.strip().split("\n")[-1]
            chosen = candidate if candidate else ""
            key_words.append(chosen)
            print(
                f"[✓] {os.path.basename(img_path)}\n{valid_labels[idx]} → {chosen}")
        except Exception as e:
            print(f"[×] Failed to extract keyword from {img_path}: {e}")

    # 保存结果
    output_path = os.path.join(output_dir, name + "_key_words.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(key_words, f, ensure_ascii=False, indent=4)
    print(f"[✓] 提取完成，已保存关键词到：{output_path}")

    # Block 更新
    bm = BlockManager()
    bid = slice.get_global_id(0)
    info = slice.get_block_info()
    info["title"]["key_word_path"] = os.path.relpath(output_path, DATASET_DIR)
    bm.update_block(bid, info)


def test_get_labels_for_thumbnails():
    name1 = "real"
    name2 = "synthetic"
    slice1 = Slice('Infographic-real')
    slice2 = Slice('Infographic-synthetic')
    output_dir = os.path.join(DATASET_DIR, 'Infographic')
    # get_labels_for_thumbnails(slice1, slice2, output_dir, name1)
    # get_labels_for_thumbnails(slice2, slice1, output_dir, name2)
    # check_labels(slice1, output_dir, name1)
    # apply_cached_thumbnails(slice1, output_dir, name1)
    # recover_labels(slice1, output_dir, name1)

    # check_labels_only(slice1, output_dir, name1)
    get_key_words(slice1, output_dir, name1)
    # get_key_words(slice2, output_dir, name2)
