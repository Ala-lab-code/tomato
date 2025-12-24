# preprocess.py
import os
import sys
import random
import shutil
import json
from pathlib import Path
from PIL import Image

def is_valid_image(img_path):
    """检查图片是否能正常打开"""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except:
        return False

def compute_class_weights(class_counts):
    """计算类别权重"""
    smooth_weights = {cls: 1.0 / (count + 0.1) for cls, count in class_counts.items()}

    weights = smooth_weights

    # 归一化到 1~2 之间
    min_w, max_w = min(weights.values()), max(weights.values())
    if max_w > min_w:
        normalized = {k: 1.0 + (v - min_w) / (max_w - min_w) for k, v in weights.items()}
    else:
        normalized = {k: 1.0 for k in weights.keys()}
    return normalized

def split_dataset(raw_data_dir, output_dir, train_ratio=0.65, val_ratio=0.15, test_ratio=0.2, min_samples_per_class=10, seed=42):
    """
    简单划分数据集，同时计算类别权重
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    splits = ["train", "val", "test"]

    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    split_info = {}
    class_counts = {}

    for class_dir in raw_data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        all_images = list(class_dir.glob("*.jpg"))
        valid_images = [img for img in all_images if is_valid_image(img)]
        if len(valid_images) < min_samples_per_class:
            print(f"类别 {class_name} 样本不足 {min_samples_per_class}，已跳过")
            continue

        random.shuffle(valid_images)
        total = len(valid_images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        train_imgs = valid_images[:train_end]
        val_imgs = valid_images[train_end:val_end]
        test_imgs = valid_images[val_end:]

        # 创建目录并复制文件
        for split_name, images in zip(splits, [train_imgs, val_imgs, test_imgs]):
            split_class_dir = output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            for img_path in images:
                shutil.copy(img_path, split_class_dir / img_path.name)

        split_info[class_name] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }
        class_counts[class_name] = total
        print(f"{class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # 计算类别权重
    class_weights = compute_class_weights(class_counts)
    print("\n类别权重 (平滑归一化):")
    for cls, w in class_weights.items():
        print(f"  {cls}: {w:.3f}")

    # 保存 split 信息
    split_info_path = output_dir / "split_metadata.json"
    metadata = {
        "split_info": split_info,
        "class_weights": class_weights
    }
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"\n数据划分完成，信息已保存到: {split_info_path}")

    return metadata

def main():

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    sys.path.append(BASE_DIR)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    split_dataset(
        raw_data_dir=RAW_DATA_DIR,
        output_dir=PROCESSED_DATA_DIR,
        train_ratio=0.65,
        val_ratio=0.15,
        test_ratio=0.2,
        min_samples_per_class=10,
        seed=42
    )

if __name__ == "__main__":
    main()
