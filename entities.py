import logging
import os
from collections import deque
from PIL import Image
from io import BytesIO
import requests
import torch
import clip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Item:
    """物品类，表示系统中的内容项"""
    def __init__(self, item_idx,item_id, name,author, categories, keywords,discrete_features,
                 continuous_features,create_time,image,description,content_feature=None,relevance_score=None):
        """
        初始化物品类
        Args:
            item_idx : 物品索引
            item_id: 物品ID
            name: 物品标题
            categories: 物品所属类目列表
            keywords: 物品包含的关键词列表
            discrete_features: 离散特征字典，如 {'category': 'A'}
            continuous_features: 连续特征字典，如 {'price': 50.0}
            create_time: 物品创建时间
            image: 物品图片url
            description: 物品文字描述
            content_feature: 基于内容的特征向量
        """
        self.item_idx=item_idx
        self.item_id = item_id
        self.name = name
        self.author=author
        self.categories = categories
        self.keywords = keywords
        self.create_time = create_time

        self.discrete_features = discrete_features or {}
        self.continuous_features = continuous_features or {}

        self.image = image
        self.description = description
        # 基于内容的向量表征
        self.content_feature = content_feature
        self.relevance_score = relevance_score

    def calculate_content_feature(self,model,preprocess):
        calculate_content_features_batch([self], model, preprocess,
                                         batch_size=1, download_workers=1)

    def __repr__(self):
        return str({'item_id':self.item_id, 'name':self.name,'keywords':self.keywords,'description':self.description,'relevance_score':self.relevance_score})


def _safe_cache_name(item):
    """生成稳定的图片缓存文件名，避免 item_id 中出现路径字符。"""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(item.item_id)) + ".img"


def _load_and_preprocess_image(item, preprocess, cache_dir, timeout, retries):
    """下载（或读取缓存）并完成 CPU 图像预处理。供线程池调用。"""
    cache_path = Path(cache_dir) / _safe_cache_name(item)
    try:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            image_bytes = cache_path.read_bytes()
        else:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            }
            last_error = None
            for attempt in range(retries + 1):
                try:
                    with requests.get(
                            item.image,
                            stream=True,
                            headers=headers,
                            timeout=timeout,
                    ) as response:
                        response.raise_for_status()
                        image_bytes = response.content
                    cache_path.write_bytes(image_bytes)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == retries:
                        raise last_error
                    time.sleep(min(2 ** attempt, 4))

        with Image.open(BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            return preprocess(rgb_image), None
    except Exception as exc:
        return None, exc


def calculate_content_features_batch(
    items,
    model,
    preprocess,
    batch_size=None,
    download_workers=None,
    cache_dir="data/image_cache",
    timeout=(10, 30),
    retries=2,
):
    """
    批量提取物品图文特征。

    图片下载和 CPU 预处理并行执行，CLIP 图像/文本编码按 batch 放到 GPU
    上执行。失败的图片使用全零图像特征，但仍保留文本特征，避免单张图片
    的网络问题中断整个离线流程。
    """
    if not items:
        return

    # CLIP 推理应尽量批量化；图片下载和 Pillow 预处理则限制并发，避免
    # CPU 过载反过来拖慢 GPU。两个参数也可以通过环境变量调节。
    if batch_size is None:
        default_batch_size = 128 if device.type == "cuda" else 16
        batch_size = int(os.getenv("RS_CLIP_BATCH_SIZE", default_batch_size))
    if download_workers is None:
        default_workers = min(8, max(2, os.cpu_count() or 4))
        download_workers = int(os.getenv("RS_IMAGE_DOWNLOAD_WORKERS", default_workers))
    batch_size = max(1, batch_size)
    download_workers = max(1, download_workers)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    total = len(items)
    image_dim = int(getattr(getattr(model, "visual", None), "output_dim", 512))
    failed = 0

    print(
        f"开始批量提取物品图文特征: total={total}, "
        f"batch_size={batch_size}, download_workers={download_workers}, device={device}",
        flush=True,
    )
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)}，显存已分配 "
            f"{torch.cuda.memory_allocated(device) / 1024 ** 2:.1f} MiB",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as executor:
        for start in range(0, total, max(1, batch_size)):
            batch_items = items[start:start + max(1, batch_size)]
            futures = [
                executor.submit(
                    _load_and_preprocess_image,
                    item,
                    preprocess,
                    cache_path,
                    timeout,
                    retries,
                )
                for item in batch_items
            ]
            prepared_images = []
            valid_indices = []
            for index, future in enumerate(futures):
                image_tensor, error = future.result()
                if error is not None:
                    failed += 1
                    logging.warning(
                        "图片处理失败，item_id=%s，使用零图像特征继续: %s",
                        batch_items[index].item_id,
                        error,
                    )
                    continue
                prepared_images.append(image_tensor)
                valid_indices.append(index)

            descriptions = [
                item.description if isinstance(item.description, str) else ""
                for item in batch_items
            ]
            text_tokens = clip.tokenize(descriptions)
            if device.type == "cuda":
                text_tokens = text_tokens.pin_memory()
            text_tokens = text_tokens.to(device, non_blocking=True)
            with torch.inference_mode():
                text_features = model.encode_text(text_tokens).float()
                image_features = torch.zeros(
                    (len(batch_items), image_dim), device=device, dtype=torch.float32
                )
                if prepared_images:
                    image_batch = torch.stack(prepared_images)
                    if device.type == "cuda":
                        image_batch = image_batch.pin_memory()
                    image_batch = image_batch.to(device, non_blocking=True)
                    encoded_images = model.encode_image(image_batch).float()
                    image_features[valid_indices] = encoded_images

                content_features = torch.cat((image_features, text_features), dim=1).cpu()

            for index, item in enumerate(batch_items):
                item.content_feature = content_features[index:index + 1]

            processed = min(start + len(batch_items), total)
            print(
                f"物品图文特征处理进度: {processed}/{total}"
                + (f"，图片失败 {failed} 张" if failed else ""),
                flush=True,
            )

    print(f"物品图文特征处理完成，图片失败 {failed} 张", flush=True)

class UserProfile:
    """用户画像类"""
    def __init__(self, user_idx,user_id, categories=None, keywords=None, discrete_features=None,
                 continuous_features=None,max_history=50):
        """
        初始化用户画像
        Args:
            user_idx: 用户索引 (0-n_users)
            user_id: 用户ID
            categories: 用户感兴趣的类目列表
            keywords: 用户感兴趣的关键词列表
            max_history: 用户历史交互队列的最大长度
            discrete_features: 离散特征字典，如 {'gender': 'M'}
            continuous_features: 连续特征字典，如 {'age': 25}
        """
        self.user_idx = user_idx
        self.user_id = user_id
        self.categories = categories  # 存储用户感兴趣的类目
        self.keywords = keywords  # 存储用户感兴趣的关键词
        self.interaction_history = deque(maxlen=max_history)  # 使用双端队列限制历史长度
        self.discrete_features = discrete_features or {}
        self.continuous_features = continuous_features or {}

    def __repr__(self):
        return str(self.__dict__)

    def add_interaction(self, item_id):
        """添加用户交互记录"""
        self.interaction_history.append(item_id)

    def get_last_n_interactions(self, n):
        """获取用户最近的n次交互记录"""
        return list(self.interaction_history)[-n:]
