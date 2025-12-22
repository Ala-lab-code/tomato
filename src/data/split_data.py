import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset_with_stratify(data_dir, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    使用分层抽样划分数据集，确保每个类别的分布与总体一致
    严格遵守 Training Set (60-70%) / Validation Set (10-20%) / Test Set (20%) 的比例规则
    """

    # 验证比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"比例总和应为1.0，但得到{train_ratio}+{val_ratio}+{test_ratio}={total_ratio}")

    # 验证比例范围
    if not 0.6 <= train_ratio <= 0.7:
        print(f"警告: 训练集比例 {train_ratio} 不在推荐范围 0.6-0.7 内")

    if not 0.1 <= val_ratio <= 0.2:
        print(f"警告: 验证集比例 {val_ratio} 不在推荐范围 0.1-0.2 内")

    if abs(test_ratio - 0.2) > 0.001:
        print(f"警告: 测试集比例 {test_ratio} 不是推荐的 0.2")

    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    # 收集所有图像路径和标签
    all_images = []
    all_labels = []
    class_names = []

    # 获取所有类别文件夹
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()  # 确保顺序一致

    print(f"找到 {len(classes)} 个类别: {classes}")

    # 统计每个类别的图像数量
    class_image_counts = {}

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)

        # 支持更多图像格式
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(image_extensions)]

        if len(images) == 0:
            print(f"警告: 类别 '{class_name}' 中没有找到图像文件!")
            continue

        for img in images:
            all_images.append(os.path.join(class_name, img))  # 保存相对路径
            all_labels.append(label)

        class_names.append(class_name)
        class_image_counts[class_name] = len(images)
        print(f"类别 '{class_name}': {len(images)} 张图片")

    if len(all_images) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到任何图像文件!")

    # 转换为numpy数组
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    print(f"\n总图片数量: {len(all_images)}")
    print(f"目标比例: 训练集 {train_ratio * 100:.0f}% | 验证集 {val_ratio * 100:.0f}% | 测试集 {test_ratio * 100:.0f}%")

    # 第一次划分：训练集 vs 验证+测试集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_state)

    for train_idx, temp_idx in sss.split(all_images, all_labels):
        train_images = all_images[train_idx]
        train_labels = all_labels[train_idx]
        temp_images = all_images[temp_idx]
        temp_labels = all_labels[temp_idx]

    # 第二次划分：验证集 vs 测试集
    val_in_temp_ratio = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (val_ratio + test_ratio),
                                  random_state=random_state)

    for val_idx, test_idx in sss2.split(temp_images, temp_labels):
        val_images = temp_images[val_idx]
        val_labels = temp_labels[val_idx]
        test_images = temp_images[test_idx]
        test_labels = temp_labels[test_idx]

    # 现在复制文件到对应目录
    def copy_files(image_paths, labels, split_name):
        """将文件复制到对应分割目录"""
        for img_path, label in zip(image_paths, labels):
            class_name = class_names[label]
            src = os.path.join(data_dir, img_path)
            dst_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(img_path))
            shutil.copy2(src, dst)

    # 复制训练集
    print("\n复制训练集...")
    copy_files(train_images, train_labels, 'train')

    # 复制验证集
    print("复制验证集...")
    copy_files(val_images, val_labels, 'val')

    # 复制测试集
    print("复制测试集...")
    copy_files(test_images, test_labels, 'test')

    # 统计信息
    print(f"\n{'=' * 60}")
    print(f"数据集划分完成!")
    print(f"总图片数量: {len(all_images)}")
    print(f"训练集: {len(train_images)} ({len(train_images) / len(all_images) * 100:.1f}%)")
    print(f"验证集: {len(val_images)} ({len(val_images) / len(all_images) * 100:.1f}%)")
    print(f"测试集: {len(test_images)} ({len(test_images) / len(all_images) * 100:.1f}%)")

    # 检查每个类别的分布
    print(f"\n分布检查:")
    print("-" * 80)
    header = f"{'类别':<40} {'训练集':<8} {'训练集%':<8} {'验证集':<8} {'验证集%':<8} {'测试集':<8} {'测试集%':<8} {'总数':<8}"
    print(header)
    print("-" * 80)

    for label, class_name in enumerate(class_names):
        train_count = np.sum(train_labels == label)
        val_count = np.sum(val_labels == label)
        test_count = np.sum(test_labels == label)
        total_count = train_count + val_count + test_count

        if total_count > 0:
            train_pct = train_count / total_count * 100
            val_pct = val_count / total_count * 100
            test_pct = test_count / total_count * 100
        else:
            train_pct = val_pct = test_pct = 0

        print(
            f"{class_name:<40} {train_count:<8} {train_pct:<8.1f} {val_count:<8} {val_pct:<8.1f} {test_count:<8} {test_pct:<8.1f} {total_count:<8}")

    print(f"{'=' * 60}")
    print(f"输出目录: {output_dir}")

    # 保存划分统计信息到文件
    stats_file = os.path.join(output_dir, "split_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("数据集划分统计信息\n")
        f.write("=" * 60 + "\n")
        f.write(f"原始数据集路径: {data_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"划分比例: 训练集={train_ratio}, 验证集={val_ratio}, 测试集={test_ratio}\n")
        f.write(f"随机种子: {random_state}\n\n")
        f.write(f"总图片数量: {len(all_images)}\n")
        f.write(f"训练集: {len(train_images)} ({len(train_images) / len(all_images) * 100:.1f}%)\n")
        f.write(f"验证集: {len(val_images)} ({len(val_images) / len(all_images) * 100:.1f}%)\n")
        f.write(f"测试集: {len(test_images)} ({len(test_images) / len(all_images) * 100:.1f}%)\n")
        f.write("\n各类别详细分布:\n")
        f.write("-" * 80 + "\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")

        for label, class_name in enumerate(class_names):
            train_count = np.sum(train_labels == label)
            val_count = np.sum(val_labels == label)
            test_count = np.sum(test_labels == label)
            total_count = train_count + val_count + test_count

            if total_count > 0:
                train_pct = train_count / total_count * 100
                val_pct = val_count / total_count * 100
                test_pct = test_count / total_count * 100
            else:
                train_pct = val_pct = test_pct = 0

            f.write(
                f"{class_name:<40} {train_count:<8} {train_pct:<8.1f} {val_count:<8} {val_pct:<8.1f} {test_count:<8} {test_pct:<8.1f} {total_count:<8}\n")

    print(f"\n统计信息已保存到: {stats_file}")

    return {
        'total_images': len(all_images),
        'train_images': len(train_images),
        'val_images': len(val_images),
        'test_images': len(test_images),
        'class_names': class_names,
        'class_image_counts': class_image_counts
    }


def check_dataset_structure(data_dir):
    """
    检查数据集结构
    """
    if not os.path.exists(data_dir):
        return False, f"路径 '{data_dir}' 不存在!"

    # 获取所有子文件夹（假设这些是类别文件夹）
    items = os.listdir(data_dir)
    class_folders = [item for item in items if os.path.isdir(os.path.join(data_dir, item))]

    if len(class_folders) == 0:
        return False, f"在 '{data_dir}' 中没有找到任何类别文件夹!"

    # 检查每个类别文件夹中是否有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
    valid_classes = []

    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(image_extensions)]

        if len(images) > 0:
            valid_classes.append((class_name, len(images)))

    if len(valid_classes) == 0:
        return False, f"在所有类别文件夹中都没有找到图像文件!"

    return True, valid_classes


def main():
    """
    主函数：执行数据集划分
    所有参数直接设定，无用户输入
    """
    print("=" * 70)
    print("番茄病害数据集划分工具")
    print("使用分层抽样方法，确保每个类别的分布与总体一致")
    print("严格遵守 Training Set (60-70%) / Validation Set (10-20%) / Test Set (20%) 规则")
    print("=" * 70)

    # ============================================================
    # 直接设定的参数（请根据实际情况修改）
    # ============================================================

    # 原始数据集路径（根据你的截图，似乎是包含10个番茄病害类别的文件夹）
    data_dir = r"C:\Users\someb\Desktop\tomato_disease_classification\data\raw"  # 修改为你的实际路径

    # 输出目录
    output_dir = r"C:\Users\someb\Desktop\tomato_disease_classification\data\splits"  # 修改为你想要的输出路径

    # 划分比例（严格遵守 Training Set (60-70%) / Validation Set (10-20%) / Test Set (20%)）
    train_ratio = 0.7  # 训练集 70%
    val_ratio = 0.1  # 验证集 10%
    test_ratio = 0.2  # 测试集 20%

    # 随机种子
    random_state = 42

    # ============================================================
    # 显示配置信息
    # ============================================================
    print("\n配置信息:")
    print("-" * 40)
    print(f"原始数据集路径: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: 训练集={train_ratio}, 验证集={val_ratio}, 测试集={test_ratio}")
    print(f"随机种子: {random_state}")

    # 检查数据集结构
    print("\n检查数据集结构...")
    is_valid, result = check_dataset_structure(data_dir)

    if not is_valid:
        print(f"错误: {result}")
        print("\n请确保数据集结构如下:")
        print("数据集文件夹/")
        print("├── 类别1/")
        print("│   ├── 图片1.jpg")
        print("│   ├── 图片2.jpg")
        print("│   └── ...")
        print("├── 类别2/")
        print("│   ├── 图片1.jpg")
        print("│   └── ...")
        print("└── ...")
        exit(1)

    # 显示找到的类别
    valid_classes = result
    print(f"找到 {len(valid_classes)} 个有效的类别:")
    total_images = 0
    for i, (class_name, count) in enumerate(valid_classes):
        print(f"  {i + 1:2d}. {class_name:<45} {count:>4} 张图片")
        total_images += count

    print(f"总计: {total_images} 张图片")

    # 检查输出目录是否已存在
    if os.path.exists(output_dir):
        print(f"\n警告: 输出目录 '{output_dir}' 已存在!")
        print("正在删除旧目录...")
        try:
            shutil.rmtree(output_dir)
            print(f"已删除旧目录: {output_dir}")
        except Exception as e:
            print(f"删除目录失败: {e}")
            exit(1)

    # 预计划分数量
    print(f"\n预计划分:")
    print(f"  训练集: {int(total_images * train_ratio)} 张")
    print(f"  验证集: {int(total_images * val_ratio)} 张")
    print(f"  测试集: {int(total_images * test_ratio)} 张")

    # 执行划分
    print("\n" + "=" * 70)
    print("开始划分数据集...")
    print("=" * 70)

    try:
        stats = split_dataset_with_stratify(
            data_dir=data_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )

        print("\n" + "=" * 70)
        print("数据集划分完成!")
        print("=" * 70)

        # 显示划分结果
        print(f"\n划分结果:")
        print(f"  训练集: {stats['train_images']} 张图片")
        print(f"  验证集: {stats['val_images']} 张图片")
        print(f"  测试集: {stats['test_images']} 张图片")
        print(f"  总计: {stats['total_images']} 张图片")

        # 显示数据集结构
        print(f"\n数据集结构:")
        print(f"  {output_dir}/")
        print(f"    ├── train/")
        print(f"    │   ├── 类别1/")
        print(f"    │   ├── 类别2/")
        print(f"    │   └── ...")
        print(f"    ├── val/")
        print(f"    │   ├── 类别1/")
        print(f"    │   ├── 类别2/")
        print(f"    │   └── ...")
        print(f"    └── test/")
        print(f"        ├── 类别1/")
        print(f"        ├── 类别2/")
        print(f"        └── ...")
        print(f"    └── split_statistics.txt  (划分统计文件)")

        print(f"\n划分后的数据集已保存到: {output_dir}")

    except Exception as e:
        print(f"\n划分过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n请检查:")
        print("  1. 数据路径是否正确")
        print("  2. 每个类别文件夹中是否有图片文件")
        print("  3. 是否有足够的磁盘空间")
        print("  4. 是否有文件权限问题")
        exit(1)


if __name__ == "__main__":
    main()