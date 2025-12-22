# features.py
import json
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops


class EnhancedFeatureExtractor:
    """增强版特征提取器，结合多种特征"""

    def __init__(self, img_size: int = 128):
        self.img_size = img_size
        self.feature_dim = None

    def extract_all_features(self, image_array: np.ndarray) -> np.ndarray:
        """提取所有特征"""
        features = []

        # 确保图像是RGB格式
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA -> RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 1:  # 灰度 -> RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:  # 灰度图
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        # 1. 颜色特征
        color_features = self._extract_color_features(image_array)
        features.extend(color_features)

        # 2. 纹理特征
        texture_features = self._extract_texture_features(image_array)
        features.extend(texture_features)

        # 3. 形状特征 (HOG)
        shape_features = self._extract_hog_features(image_array)
        features.extend(shape_features)

        # 4. 边缘特征
        edge_features = self._extract_edge_features(image_array)
        features.extend(edge_features)

        # 5. 统计特征
        stat_features = self._extract_statistical_features(image_array)
        features.extend(stat_features)

        return np.array(features, dtype=np.float32)

    def _extract_color_features(self, image_array: np.ndarray):
        """提取颜色特征"""
        features = []

        # RGB颜色直方图
        for i in range(3):
            hist = cv2.calcHist([image_array], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        # HSV颜色空间
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)

        # 颜色矩
        for i in range(3):
            channel = image_array[:, :, i].flatten()
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(skew(channel))

        # 颜色相关性
        r_g_corr = np.corrcoef(image_array[:, :, 0].flatten(), image_array[:, :, 1].flatten())[0, 1]
        r_b_corr = np.corrcoef(image_array[:, :, 0].flatten(), image_array[:, :, 2].flatten())[0, 1]
        g_b_corr = np.corrcoef(image_array[:, :, 1].flatten(), image_array[:, :, 2].flatten())[0, 1]
        features.extend([r_g_corr, r_b_corr, g_b_corr])

        return features

    def _extract_texture_features(self, image_array: np.ndarray):
        """提取纹理特征"""
        features = []

        # 转换为灰度图
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # LBP特征
        radius = 3
        n_points = 8 * radius
        try:
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)))
            lbp_hist = lbp_hist.astype("float")
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist.tolist())
        except:
            features.extend([0.0] * 59)

        # 灰度共生矩阵特征
        try:
            glcm = graycomatrix(gray.astype(np.uint8), distances=[1, 3],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                features.extend(prop_values.flatten().tolist())
        except:
            features.extend([0.0] * (len(properties) * 2 * 4))

        # 梯度特征
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))

        return features

    def _extract_hog_features(self, image_array: np.ndarray):
        """提取HOG特征"""
        features = []

        # 转换为灰度图
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # 计算HOG特征
        try:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), block_norm='L2-Hys',
                               transform_sqrt=True, feature_vector=True)
            features.extend(hog_features.tolist())
        except:
            features.extend([0.0] * 324)

        return features

    def _extract_edge_features(self, image_array: np.ndarray):
        """提取边缘特征"""
        features = []

        # 转换为灰度图
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Canny边缘检测
        edges_low = cv2.Canny(gray, 30, 100)
        edges_high = cv2.Canny(gray, 100, 200)

        features.append(np.sum(edges_low) / 255.0)
        features.append(np.sum(edges_high) / 255.0)
        features.append(np.mean(edges_low))
        features.append(np.mean(edges_high))

        # 计算轮廓特征
        contours_low, _ = cv2.findContours(edges_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_low) > 0:
            areas = [cv2.contourArea(c) for c in contours_low]
            features.append(len(contours_low))
            features.append(np.mean(areas))
            features.append(np.std(areas))
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _extract_statistical_features(self, image_array: np.ndarray):
        """提取统计特征"""
        features = []

        # 转换为灰度图
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # 基本统计
        gray_flat = gray.flatten()
        features.append(np.mean(gray_flat))
        features.append(np.std(gray_flat))
        features.append(np.min(gray_flat))
        features.append(np.max(gray_flat))

        # 分位数
        features.append(np.percentile(gray_flat, 25))
        features.append(np.percentile(gray_flat, 50))
        features.append(np.percentile(gray_flat, 75))

        # 偏度和峰度
        features.append(skew(gray_flat))
        features.append(kurtosis(gray_flat))

        # 熵
        hist, _ = np.histogram(gray_flat, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features.append(entropy)

        return features

    def get_feature_dimension(self):
        """获取特征维度"""
        if self.feature_dim is None:
            test_image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            features = self.extract_all_features(test_image)
            self.feature_dim = len(features)
        return self.feature_dim


class DataLoaderSimple:
    """简单的数据加载器，集成类别权重"""

    def __init__(self, data, labels, class_weights=None, batch_size=32, shuffle=True):
        """
        初始化简单的数据加载器

        Args:
            data: 特征数据，numpy数组
            labels: 标签数据，numpy数组
            class_weights: 类别权重字典，键为类别索引，值为权重
            batch_size: 批次大小
            shuffle: 是否打乱数据
        """
        self.data = data
        self.labels = labels
        self.class_weights = class_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_batch = 0

        # 计算样本权重
        self.sample_weights = None
        if class_weights is not None:
            self.sample_weights = np.array([class_weights[label] for label in labels])

        # 如果需要打乱，创建索引并打乱
        self.indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        batch_indices = self.indices[start_idx:end_idx]
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]

        batch_weights = None
        if self.sample_weights is not None:
            batch_weights = self.sample_weights[batch_indices]

        self.current_batch += 1

        if batch_weights is not None:
            return batch_data, batch_labels, batch_weights
        else:
            return batch_data, batch_labels

    def __len__(self):
        return self.num_batches

    def get_class_distribution(self):
        """获取类别分布"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def load_and_preprocess_data(data_dir, mode='train', img_size=128,
                             class_weights=None,
                             class_order=None,
                             expected_counts=None,
                             feature_extractor=None):
    """
    加载和预处理数据，返回特征和标签

    Args:
        data_dir: 数据目录
        mode: 数据集模式 ('train', 'val', 'test')
        img_size: 图像尺寸
        class_weights: 类别权重字典
        class_order: 类别顺序（与划分阶段保持一致）
        expected_counts: 期望的样本数（字典，键为类别名，值为期望数量）
        feature_extractor: 特征提取器

    Returns:
        X: 特征数组
        y: 标签数组
        sample_weights: 样本权重数组（如果有）
        actual_class_names: 实际加载的类别名称列表
    """
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor(img_size=img_size)

    # 构建完整的数据路径
    full_data_dir = os.path.join(data_dir, mode)

    if not os.path.exists(full_data_dir):
        print(f"警告: 数据集路径不存在: {full_data_dir}")
        return np.array([]), np.array([]), np.array([]), []

    # 获取所有类别 - 使用指定的类别顺序
    if class_order is not None:
        actual_class_names = []
        for class_name in class_order:
            class_dir = os.path.join(full_data_dir, class_name)
            if os.path.isdir(class_dir):
                actual_class_names.append(class_name)
            else:
                print(f"警告: 类别目录 {class_dir} 不存在，跳过 {class_name}")
    else:
        # 按照目录中的顺序（不是字母顺序）
        actual_class_names = []
        for item in os.listdir(full_data_dir):
            item_path = os.path.join(full_data_dir, item)
            if os.path.isdir(item_path):
                actual_class_names.append(item)

    if len(actual_class_names) == 0:
        print(f"警告: 在 {full_data_dir} 中没有找到任何类别目录!")
        return np.array([]), np.array([]), np.array([]), []

    # 创建类别到索引的映射
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(actual_class_names)}

    # 收集所有样本
    print(f"\n加载 {mode} 数据集...")
    all_samples = []
    class_sample_counts = defaultdict(int)

    for class_name in tqdm(actual_class_names, desc=f"加载{mode}类别"):
        class_dir = os.path.join(full_data_dir, class_name)

        # 获取所有图片文件（按文件名排序以确保一致性）
        image_files = []
        for f in sorted(os.listdir(class_dir)):
            f_lower = f.lower()
            if f_lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
                image_files.append(f)

        # 验证数量（如果有期望值）
        if expected_counts and class_name in expected_counts:
            expected = expected_counts[class_name]
            if len(image_files) != expected:
                print(f"⚠️  {class_name}: 期望 {expected} 个文件, 实际 {len(image_files)} 个文件, "
                      f"差异: {len(image_files) - expected}")

        label = class_to_idx[class_name]
        class_sample_counts[class_name] = len(image_files)

        # 添加样本路径和标签
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            all_samples.append((img_path, label, class_name))

    if len(all_samples) == 0:
        print(f"警告: 在 {full_data_dir} 中没有找到任何图像文件!")
        return np.array([]), np.array([]), np.array([]), actual_class_names

    # 提取特征
    print(f"提取特征...")
    X_list = []
    y_list = []
    failed_samples = []

    for img_path, label, class_name in tqdm(all_samples, desc="提取特征"):
        try:
            # 加载图像
            with Image.open(img_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 调整大小
                img = img.resize((img_size, img_size))

                # 转换为numpy数组
                img_array = np.array(img)

                # 提取特征
                features = feature_extractor.extract_all_features(img_array)

                X_list.append(features)
                y_list.append(label)

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            failed_samples.append(img_path)

    X = np.array(X_list)
    y = np.array(y_list)

    if failed_samples:
        print(f"⚠️  {len(failed_samples)} 个样本处理失败")

    # 计算样本权重
    sample_weights = None
    if class_weights is not None and len(y) > 0:
        # 将类别名称权重转换为类别索引权重
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        weight_array = []
        for label in y:
            class_name = idx_to_class.get(label, None)
            if class_name and class_name in class_weights:
                weight_array.append(class_weights[class_name])
            else:
                weight_array.append(1.0)
        sample_weights = np.array(weight_array, dtype=np.float32)

    # 统计信息
    print(f"\n{'=' * 50}")
    print(f"增强特征数据集 '{mode}' 加载完成")
    print(f"{'=' * 50}")
    print(f"总图片数: {len(X):,}")
    print(f"类别数: {len(actual_class_names)}")
    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"处理失败: {len(failed_samples)}")

    # 显示类别分布
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n类别分布:")
        total_difference = 0

        for label, count in zip(unique, counts):
            class_name = actual_class_names[label]
            actual_count = count
            expected_count = expected_counts.get(class_name, 0) if expected_counts else 0

            if expected_counts:
                difference = actual_count - expected_count
                total_difference += abs(difference)
                diff_str = f" (期望: {expected_count}, 差异: {difference:+d})"
            else:
                diff_str = ""

            weight_str = f", 权重: {class_weights.get(class_name, 1.0):.3f}" if class_weights else ""
            print(f"  {class_name}: {actual_count}个样本{weight_str}{diff_str}")

        if expected_counts and total_difference > 0:
            print(f"\n⚠️  总差异: {total_difference} 个样本")

    return X, y, sample_weights, actual_class_names


def create_dataloaders(data_dir, img_size=128,
                       class_weights=None,
                       split_metadata=None,
                       class_order_file=None,
                       batch_size=32):
    """
    创建数据加载器（与划分模块保持一致）

    Args:
        data_dir: 数据目录
        img_size: 图像尺寸
        class_weights: 类别权重字典
        split_metadata: 划分元数据（包含统计信息）
        class_order_file: 类别顺序文件路径
        batch_size: 批次大小

    Returns:
        train_loader, val_loader, test_loader, class_names, feature_dim, class_to_idx, sample_weights_info
    """
    import json

    print("=" * 60)
    print("创建数据加载器（与划分模块保持一致）")
    print("=" * 60)

    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"批次大小: {batch_size}")

    # 初始化特征提取器
    feature_extractor = EnhancedFeatureExtractor(img_size=img_size)
    feature_dim = feature_extractor.get_feature_dimension()

    # 加载类别顺序
    class_order = None
    if class_order_file and os.path.exists(class_order_file):
        with open(class_order_file, 'r') as f:
            class_order = [line.strip() for line in f if line.strip()]
        print(f"从文件加载类别顺序: {len(class_order)} 个类别")
    elif split_metadata and 'filtered_classes' in split_metadata:
        class_order = split_metadata['filtered_classes']
        print(f"从元数据加载类别顺序: {len(class_order)} 个类别")

    # 准备期望的样本数（用于验证）
    train_expected = None
    val_expected = None
    test_expected = None

    if split_metadata:
        if 'train_counts' in split_metadata:
            train_expected = split_metadata['train_counts']
        if 'val_counts' in split_metadata:
            val_expected = split_metadata['val_counts']
        if 'test_counts' in split_metadata:
            test_expected = split_metadata['test_counts']

    # 加载训练数据
    X_train, y_train, train_weights, train_class_names = load_and_preprocess_data(
        data_dir=data_dir,
        mode='train',
        img_size=img_size,
        class_weights=class_weights,
        class_order=class_order,
        expected_counts=train_expected,
        feature_extractor=feature_extractor
    )

    # 确保类别名称一致
    if len(train_class_names) == 0:
        raise ValueError("训练集没有加载到任何数据!")

    class_names = train_class_names
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    # 加载验证数据
    X_val, y_val, val_weights, val_class_names = load_and_preprocess_data(
        data_dir=data_dir,
        mode='val',
        img_size=img_size,
        class_weights=class_weights,
        class_order=class_order,
        expected_counts=val_expected,
        feature_extractor=feature_extractor
    )

    # 加载测试数据
    X_test, y_test, test_weights, test_class_names = load_and_preprocess_data(
        data_dir=data_dir,
        mode='test',
        img_size=img_size,
        class_weights=class_weights,
        class_order=class_order,
        expected_counts=test_expected,
        feature_extractor=feature_extractor
    )

    # 验证类别名称一致性
    if (len(val_class_names) > 0 and val_class_names != class_names) or \
            (len(test_class_names) > 0 and test_class_names != class_names):
        print("⚠️  警告: 不同数据集的类别名称不一致!")
        print(f"训练集类别: {class_names}")
        print(f"验证集类别: {val_class_names}")
        print(f"测试集类别: {test_class_names}")

    # 创建数据加载器
    train_loader = DataLoaderSimple(
        X_train, y_train,
        class_weights=None,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoaderSimple(
        X_val, y_val,
        class_weights=None,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoaderSimple(
        X_test, y_test,
        class_weights=None,
        batch_size=batch_size,
        shuffle=False
    )

    # 计算实际比例
    total_samples = len(X_train) + len(X_val) + len(X_test)

    print(f"\n数据集统计:")
    print(f"  训练集: {len(X_train):,} 张图片 ({len(X_train) / total_samples * 100:.1f}%)")
    print(f"  验证集: {len(X_val):,} 张图片 ({len(X_val) / total_samples * 100:.1f}%)")
    print(f"  测试集: {len(X_test):,} 张图片 ({len(X_test) / total_samples * 100:.1f}%)")
    print(f"  总计: {total_samples:,} 张图片")
    print(f"  特征维度: {feature_dim}")
    print(f"  类别数: {len(class_names)}")

    # 与划分阶段对比（如果有元数据）
    if split_metadata:
        expected_train = split_metadata.get('train_samples', 0)
        expected_val = split_metadata.get('val_samples', 0)
        expected_test = split_metadata.get('test_samples', 0)
        expected_total = split_metadata.get('total_samples', 0)

        print(f"\n与划分阶段对比:")
        print(f"  训练集: {len(X_train):,} vs {expected_train:,} (差异: {len(X_train) - expected_train:+,d})")
        print(f"  验证集: {len(X_val):,} vs {expected_val:,} (差异: {len(X_val) - expected_val:+,d})")
        print(f"  测试集: {len(X_test):,} vs {expected_test:,} (差异: {len(X_test) - expected_test:+,d})")
        print(f"  总计: {total_samples:,} vs {expected_total:,} (差异: {total_samples - expected_total:+,d})")

        # 计算整体差异率
        if expected_total > 0:
            total_diff_rate = abs(total_samples - expected_total) / expected_total * 100
            if total_diff_rate > 1.0:
                print(f"⚠️  警告: 总样本差异率 {total_diff_rate:.2f}% > 1%")
            else:
                print(f"✅ 总样本差异率 {total_diff_rate:.2f}% < 1%，一致性良好")

    # 返回样本权重
    sample_weights_info = {
        'train_weights': train_weights,
        'val_weights': val_weights,
        'test_weights': test_weights
    }

    return train_loader, val_loader, test_loader, class_names, feature_dim, class_to_idx, sample_weights_info


def main():
    """主函数"""
    # 数据路径
    PROCESSED_DATA_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\processed"

    # 加载划分元数据
    metadata_file = os.path.join(PROCESSED_DATA_DIR, "split_metadata.json")
    split_metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            split_metadata = json.load(f)
        print(f"已加载划分元数据")

    # 加载类别权重
    weights_file = os.path.join(PROCESSED_DATA_DIR, "class_weights.npy")
    class_weights = {}
    if os.path.exists(weights_file):
        class_weights = np.load(weights_file, allow_pickle=True).item()
        print(f"已加载类别权重: {len(class_weights)} 个类别")

    # 加载类别顺序
    class_order_file = os.path.join(PROCESSED_DATA_DIR, "class_order.txt")

    # 创建数据加载器
    train_loader, val_loader, test_loader, loaded_class_names, feature_dim, class_to_idx, sample_weights_info = create_dataloaders(
        data_dir=PROCESSED_DATA_DIR,
        img_size=128,
        class_weights=class_weights,
        split_metadata=split_metadata,
        class_order_file=class_order_file,
        batch_size=32
    )

    print(f"\n{'=' * 60}")
    print("✓ 特征提取和数据加载完成!")
    print(f"{'=' * 60}")
    print(f"  特征维度: {feature_dim}")
    print(f"  类别数: {len(loaded_class_names)}")

    # 保存特征提取器和类别映射信息
    feature_info = {
        'feature_dim': feature_dim,
        'class_to_idx': class_to_idx,
        'class_names': loaded_class_names,
        'img_size': 128
    }

    feature_info_file = os.path.join(PROCESSED_DATA_DIR, "feature_info.pkl")
    with open(feature_info_file, 'wb') as f:
        pickle.dump(feature_info, f)
    print(f"特征信息已保存到: {feature_info_file}")

    return train_loader, val_loader, test_loader, loaded_class_names, feature_dim, class_to_idx, sample_weights_info


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names, feature_dim, class_to_idx, sample_weights_info = main()