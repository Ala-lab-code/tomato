# preprocess.py
import os
import sys
import json
import random
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def setup_chinese_font():
    """配置中文字体显示"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun',
                         'FangSong', 'STXihei', 'STKaiti', 'STSong']

        system_fonts = [f.name for f in fm.fontManager.ttflist]
        available_fonts = []
        for font in chinese_fonts:
            if any(font.lower() in f.lower() for f in system_fonts):
                available_fonts.append(font)

        if available_fonts:
            plt.rcParams['font.sans-serif'] = available_fonts
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 已设置中文字体: {available_fonts[0]}")
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠️ 未找到中文字体，使用默认字体")
    except Exception as e:
        print(f"字体设置失败: {e}")


def set_random_seed(seed=42):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ 随机种子已设置为: {seed}")


def check_dataset_structure(data_dir: str):
    """检查数据集结构并返回有效的类别"""
    if not os.path.exists(data_dir):
        return False, f"路径 '{data_dir}' 不存在!"

    items = os.listdir(data_dir)
    class_folders = []

    for item in items:
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_folders.append(item)

    if len(class_folders) == 0:
        return False, f"在 '{data_dir}' 中没有找到任何类别文件夹!"

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp')
    valid_classes = []

    for class_name in class_folders:
        class_path = os.path.join(data_dir, class_name)
        file_count = 0

        try:
            all_files = os.listdir(class_path)
            for f in all_files:
                if f.lower().endswith(image_extensions):
                    file_count += 1
        except Exception as e:
            print(f"检查类别 {class_name} 时出错: {e}")
            continue

        if file_count > 0:
            valid_classes.append((class_name, file_count))
        else:
            print(f"警告: 类别 '{class_name}' 中没有找到图像文件!")

    if len(valid_classes) == 0:
        return False, f"在所有类别文件夹中都没有找到图像文件!"

    return True, valid_classes


def compute_class_weights(class_counts):
    """
    计算类别权重

    Args:
        class_counts: 字典，键为类别名，值为样本数量

    Returns:
        class_weights: 字典，键为类别名，值为权重
    """
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)

    # 方法1: 基于频率的权重（样本数越少权重越高）
    freq_weights = {}
    for class_name, count in class_counts.items():
        freq_weights[class_name] = total_samples / (n_classes * count)

    # 方法2: 基于中位数的权重（更稳健）
    median_samples = np.median(list(class_counts.values()))
    median_weights = {}
    for class_name, count in class_counts.items():
        median_weights[class_name] = median_samples / count

    # 方法3: 平滑权重（避免极端权重）
    smooth_weights = {}
    alpha = 0.1  # 平滑参数
    for class_name, count in class_counts.items():
        smooth_weights[class_name] = 1.0 / (count + alpha)

    # 归一化权重
    def normalize_weights(weights):
        min_w = min(weights.values())
        max_w = max(weights.values())
        if max_w > min_w:
            return {k: 1.0 + (w - min_w) / (max_w - min_w) for k, w in weights.items()}
        return {k: 1.0 for k in weights.keys()}

    return {
        'freq': normalize_weights(freq_weights),
        'median': normalize_weights(median_weights),
        'smooth': normalize_weights(smooth_weights)
    }


def verify_dataset_consistency(processed_dir, stats):
    """
    验证划分后的数据集与划分统计是否一致
    """
    print("=" * 60)
    print("验证数据集一致性")
    print("=" * 60)

    actual_counts = defaultdict(lambda: defaultdict(int))
    issues = []

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_dir, split)

        if not os.path.exists(split_dir):
            issues.append(f"分割目录不存在: {split_dir}")
            continue

        # 获取实际类别
        actual_classes = [d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d))]

        # 验证每个类别
        for class_name in stats['filtered_classes']:
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.exists(class_dir):
                issues.append(f"类别目录不存在: {class_dir}")
                continue

            # 统计实际图像文件
            actual_files = []
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                    actual_files.append(f)

            actual_count = len(actual_files)
            expected_count = 0

            if split == 'train':
                expected_count = stats['train_counts'][class_name]
            elif split == 'val':
                expected_count = stats['val_counts'][class_name]
            elif split == 'test':
                expected_count = stats['test_counts'][class_name]

            actual_counts[split][class_name] = actual_count

            if actual_count != expected_count:
                issues.append(
                    f"{split}/{class_name}: 期望 {expected_count}, 实际 {actual_count}, "
                    f"差异 {actual_count - expected_count}"
                )

    # 输出验证结果
    print("\n验证结果:")
    if issues:
        print(f"⚠️ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 数据集一致，所有文件数量匹配")

    # 汇总统计
    total_actual = sum(sum(counts.values()) for counts in actual_counts.values())
    print(f"\n实际文件总数: {total_actual:,}")
    print(f"期望文件总数: {stats['total_samples']:,}")

    return actual_counts, issues


def stratified_split_dataset(data_dir, output_dir, class_names,
                             train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                             min_samples_per_class=10,
                             class_weights_method='smooth',
                             random_state=42):
    """
    带类别权重和分层抽样的数据集划分

    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        class_names: 类别列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        min_samples_per_class: 每个类别最小样本数（少于这个数的类别会被过滤）
        class_weights_method: 类别权重计算方法 ('freq', 'median', 'smooth')
        random_state: 随机种子

    Returns:
        划分统计信息和类别权重
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        raise ValueError(f"比例之和应为1.0，当前为{total_ratio}")

    print(
        f"数据集划分比例: 训练集={train_ratio * 100:.1f}%, 验证集={val_ratio * 100:.1f}%, 测试集={test_ratio * 100:.1f}%")

    # 设置随机种子
    random.seed(random_state)
    np.random.seed(random_state)

    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

    # 统计信息
    stats = {
        'class_counts': defaultdict(int),
        'train_counts': defaultdict(int),
        'val_counts': defaultdict(int),
        'test_counts': defaultdict(int),
        'total_samples': 0,
        'class_weights': {}
    }

    # 存储所有文件路径
    all_files_by_class = defaultdict(list)
    filtered_classes = []

    print("收集所有图像文件并过滤小样本类别...")
    for class_name in tqdm(class_names):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.exists(class_path):
            print(f"警告: 类别目录 {class_path} 不存在，跳过")
            continue

        # 获取所有图片文件
        image_files = []
        for f in os.listdir(class_path):
            f_lower = f.lower()
            if f_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                image_files.append(f)

        if len(image_files) >= min_samples_per_class:
            all_files_by_class[class_name] = image_files
            stats['class_counts'][class_name] = len(image_files)
            stats['total_samples'] += len(image_files)
            filtered_classes.append(class_name)
        else:
            print(f"警告: 类别 {class_name} 只有 {len(image_files)} 个样本，小于最小阈值 {min_samples_per_class}，跳过")

    print(f"\n过滤后有效类别数: {len(filtered_classes)} / {len(class_names)}")
    print(f"总样本数: {stats['total_samples']}")

    if len(filtered_classes) == 0:
        raise ValueError("没有足够的有效类别进行数据集划分")

    # 计算类别权重
    class_weights = compute_class_weights(stats['class_counts'])
    stats['class_weights'] = class_weights[class_weights_method]

    print(f"\n类别权重 ({class_weights_method}方法):")
    for class_name in filtered_classes:
        weight = stats['class_weights'].get(class_name, 1.0)
        count = stats['class_counts'][class_name]
        print(f"  {class_name}: {count}个样本, 权重={weight:.3f}")

    # 进行分层划分
    print(f"\n进行分层数据集划分...")

    for class_name in tqdm(filtered_classes, desc="划分数据集"):
        image_files = all_files_by_class[class_name]
        n_total = len(image_files)

        # 随机打乱
        random.shuffle(image_files)

        # 计算划分数量
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # 调整以确保每个集合至少有一个样本
        if n_train == 0:
            n_train = 1
            n_val = min(1, n_total - 1)
            n_test = n_total - 2
        elif n_val == 0:
            n_val = 1
            n_test = n_total - n_train - 1
        elif n_test == 0:
            n_test = 1
            n_train = n_total - n_val - 1

        # 确保非负
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        n_test = max(0, n_total - n_train - n_val)

        # 划分文件
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # 更新统计
        stats['train_counts'][class_name] = len(train_files)
        stats['val_counts'][class_name] = len(val_files)
        stats['test_counts'][class_name] = len(test_files)

        # 复制文件
        for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_dir = os.path.join(output_dir, split_name, class_name)

            for img_file in split_files:
                src_path = os.path.join(data_dir, class_name, img_file)
                dst_path = os.path.join(split_dir, img_file)

                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"复制文件 {src_path} 到 {dst_path} 失败: {e}")

        if len(filtered_classes) <= 10:  # 只显示前10个类别的详细信息
            print(f"  {class_name}: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")

    # 汇总统计
    total_train = sum(stats['train_counts'].values())
    total_val = sum(stats['val_counts'].values())
    total_test = sum(stats['test_counts'].values())

    result = {
        'total_samples': stats['total_samples'],
        'train_samples': total_train,
        'val_samples': total_val,
        'test_samples': total_test,
        'class_counts': dict(stats['class_counts']),
        'train_counts': dict(stats['train_counts']),
        'val_counts': dict(stats['val_counts']),
        'test_counts': dict(stats['test_counts']),
        'class_weights': stats['class_weights'],
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'filtered_classes': filtered_classes,
        'class_weights_method': class_weights_method,
        'min_samples_per_class': min_samples_per_class,
        'random_state': random_state
    }

    return result


def main():
    """主函数"""
    # 设置字体和随机种子
    setup_chinese_font()
    set_random_seed(42)

    # 数据路径设置
    RAW_DATA_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\raw"
    PROCESSED_DATA_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\processed"
    RESULTS_DIR = "./improved_baseline_results"
    MODELS_DIR = "./improved_models"

    # 创建必要的目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"原始数据路径: {RAW_DATA_DIR}")
    print(f"处理数据路径: {PROCESSED_DATA_DIR}")
    print(f"结果保存路径: {RESULTS_DIR}")
    print(f"模型保存路径: {MODELS_DIR}")

    # 检查原始数据集
    if os.path.exists(RAW_DATA_DIR):
        is_valid, result = check_dataset_structure(RAW_DATA_DIR)

        if is_valid:
            valid_classes = result
            class_names = [c[0] for c in valid_classes]

            print(f"\n数据集检查通过!")
            print(f"找到 {len(valid_classes)} 个类别:")

            # 显示类别统计
            class_df = pd.DataFrame(valid_classes, columns=['Class', 'Count'])
            print(f"\n类别统计:")
            print(class_df.sort_values('Count', ascending=False).to_string(index=False))

            total_images = class_df['Count'].sum()
            print(f"\n总图片数: {total_images}")
            print(f"平均每类图片数: {total_images / len(valid_classes):.1f}")
        else:
            print(f"数据集检查失败: {result}")
            class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                           'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                           'Tomato_Spider_mites_Two_spotted_spider_mite',
                           'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
                           'Tomato__Tomato_YellowLeaf__Curl_Virus',
                           'Tomato_healthy']
    else:
        print(f"警告: 原始数据路径 '{RAW_DATA_DIR}' 不存在!")
        class_names = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
                       'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
                       'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
                       'Tomato_healthy']

    # 划分数据集
    print("\n开始分层划分数据集（带类别权重）...")

    # 设置划分参数
    train_ratio = 0.7  # 70% 训练集
    val_ratio = 0.1  # 10% 验证集
    test_ratio = 0.2  # 20% 测试集

    stats = stratified_split_dataset(
        data_dir=RAW_DATA_DIR,
        output_dir=PROCESSED_DATA_DIR,
        class_names=class_names,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_samples_per_class=10,
        class_weights_method='smooth',
        random_state=42
    )

    print(f"\n✓ 数据集划分完成!")
    print(f"总样本数: {stats['total_samples']:,}")
    print(f"训练集: {stats['train_samples']:,} ({stats['train_samples'] / stats['total_samples'] * 100:.1f}%)")
    print(f"验证集: {stats['val_samples']:,} ({stats['val_samples'] / stats['total_samples'] * 100:.1f}%)")
    print(f"测试集: {stats['test_samples']:,} ({stats['test_samples'] / stats['total_samples'] * 100:.1f}%)")
    print(f"有效类别数: {len(stats['filtered_classes'])}")

    # 保存类别权重到文件
    weights_file = os.path.join(PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(weights_file, stats['class_weights'])
    print(f"\n类别权重已保存到: {weights_file}")

    # 保存划分元数据到文件
    metadata_file = os.path.join(PROCESSED_DATA_DIR, "split_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"划分元数据已保存到: {metadata_file}")

    # 保存类别顺序到文件
    class_order_file = os.path.join(PROCESSED_DATA_DIR, "class_order.txt")
    with open(class_order_file, 'w') as f:
        for class_name in stats['filtered_classes']:
            f.write(f"{class_name}\n")
    print(f"类别顺序已保存到: {class_order_file}")

    # 验证数据集一致性
    print("\n验证划分后数据集的一致性...")
    actual_counts, issues = verify_dataset_consistency(PROCESSED_DATA_DIR, stats)

    print(f"\n预处理完成！处理后的数据保存在: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()