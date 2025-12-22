import os
import random
import shutil
import json
from pathlib import Path
from PIL import Image  # 用来检测图片是否能打开


def is_valid_image(img_path):
    """
    检查图片是否能正常打开
    """
    try:
        with Image.open(img_path) as img:
            img.verify()  # verify 会检查图片文件完整性
        return True
    except:
        return False


def split_dataset(raw_data_dir, output_dir, train_ratio=0.65, val_ratio=0.15, test_ratio=0.2, seed=42):
    """
    将原始 PlantVillage 番茄数据集划分为 train / val / test
    并在切分前进行最小必要的数据清理（删除损坏图片）
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)  # 固定随机种子，保证划分可复现

    raw_data_dir = Path(raw_data_dir)
    output_dir = Path(output_dir)
    splits = ["train", "val", "test"]

    # 创建输出目录
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    split_info = {}

    # 遍历每个类别文件夹
    for class_dir in raw_data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # 获取所有 jpg 文件，并进行最小清理
        all_images = list(class_dir.glob("*.jpg"))
        valid_images = [img for img in all_images if is_valid_image(img)]

        # 如果有损坏图片，打印提示
        num_invalid = len(all_images) - len(valid_images)
        if num_invalid > 0:
            print(f"[Warning] {class_name} 有 {num_invalid} 张损坏图片已被移除")

        images = valid_images
        random.shuffle(images)  # 随机打乱顺序

        # 切分训练集、验证集、测试集
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        split_info[class_name] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }

        # 创建类别目录
        for split in splits:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        # 拷贝文件
        for img in train_imgs:
            shutil.copy(img, output_dir / "train" / class_name / img.name)
        for img in val_imgs:
            shutil.copy(img, output_dir / "val" / class_name / img.name)
        for img in test_imgs:
            shutil.copy(img, output_dir / "test" / class_name / img.name)

        print(f"{class_name}: "
              f"train={len(train_imgs)}, "
              f"val={len(val_imgs)}, "
              f"test={len(test_imgs)}")

    # 保存划分信息
    split_info_path = output_dir.parent / "splits" / "split_info.json"
    split_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=4)

    print("\nDataset split completed!")
    print(f"Split info saved to: {split_info_path}")


if __name__ == "__main__":
    split_dataset(
        raw_data_dir="../data/raw",
        output_dir="../data/processed"
    )

