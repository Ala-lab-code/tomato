# features.py
import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops


class EnhancedFeatureExtractor:
    """增强版特征提取器，结合多种特征"""

    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        self.feature_dim = None

    def extract_all_features(self, image_array: np.ndarray) -> np.ndarray:
        features = []

        # 保证RGB格式
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

        features.extend(self._extract_color_features(image_array))
        features.extend(self._extract_texture_features(image_array))
        features.extend(self._extract_hog_features(image_array))
        features.extend(self._extract_edge_features(image_array))
        features.extend(self._extract_statistical_features(image_array))

        return np.array(features, dtype=np.float32)

    def _extract_color_features(self, image_array: np.ndarray):
        features = []
        for i in range(3):
            hist = cv2.calcHist([image_array], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        for i in range(3):
            channel = image_array[:, :, i].flatten()
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(skew(channel))
        r_g_corr = np.corrcoef(image_array[:, :, 0].flatten(), image_array[:, :, 1].flatten())[0, 1]
        r_b_corr = np.corrcoef(image_array[:, :, 0].flatten(), image_array[:, :, 2].flatten())[0, 1]
        g_b_corr = np.corrcoef(image_array[:, :, 1].flatten(), image_array[:, :, 2].flatten())[0, 1]
        features.extend([r_g_corr, r_b_corr, g_b_corr])
        return features

    def _extract_texture_features(self, image_array: np.ndarray):
        features = []
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        radius = 3
        n_points = 8 * radius
        try:
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=int(lbp.max() + 1), range=(0, int(lbp.max() + 1)))
            lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
            features.extend(lbp_hist.tolist())
        except:
            features.extend([0.0] * 59)
        try:
            glcm = graycomatrix(gray.astype(np.uint8), distances=[1, 3],
                                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                features.extend(prop_values.flatten().tolist())
        except:
            features.extend([0.0] * (len(properties) * 2 * 4))
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        return features

    def _extract_hog_features(self, image_array: np.ndarray):
        features = []
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        try:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), block_norm='L2-Hys',
                               transform_sqrt=True, feature_vector=True)
            features.extend(hog_features.tolist())
        except:
            features.extend([0.0] * 324)
        return features

    def _extract_edge_features(self, image_array: np.ndarray):
        features = []
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges_low = cv2.Canny(gray, 30, 100)
        edges_high = cv2.Canny(gray, 100, 200)
        features.append(np.sum(edges_low) / 255.0)
        features.append(np.sum(edges_high) / 255.0)
        features.append(np.mean(edges_low))
        features.append(np.mean(edges_high))
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
        features = []
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        gray_flat = gray.flatten()
        features.append(np.mean(gray_flat))
        features.append(np.std(gray_flat))
        features.append(np.min(gray_flat))
        features.append(np.max(gray_flat))
        features.append(np.percentile(gray_flat, 25))
        features.append(np.percentile(gray_flat, 50))
        features.append(np.percentile(gray_flat, 75))
        features.append(skew(gray_flat))
        features.append(kurtosis(gray_flat))
        hist, _ = np.histogram(gray_flat, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features.append(entropy)
        return features

    def get_feature_dimension(self):
        if self.feature_dim is None:
            test_image = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
            features = self.extract_all_features(test_image)
            self.feature_dim = len(features)
        return self.feature_dim


class DataLoaderSimple:
    """简单的数据加载器"""

    def __init__(self, data, labels, sample_weights=None, batch_size=32, shuffle=True):
        self.data = data
        self.labels = labels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.current_batch = 0

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
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, self.num_samples)
        idx = self.indices[start:end]
        batch_data = self.data[idx]
        batch_labels = self.labels[idx]
        batch_weights = self.sample_weights[idx] if self.sample_weights is not None else None
        self.current_batch += 1
        return (batch_data, batch_labels, batch_weights) if batch_weights is not None else (batch_data, batch_labels)

    def __len__(self):
        return self.num_batches


def load_and_preprocess_data(data_dir, mode='train', img_size=224,
                             class_weights=None, split_info=None,
                             feature_extractor=None):
    """加载图像并提取特征，不再内部创建 feature_extractor"""
    if feature_extractor is None:
        feature_extractor = EnhancedFeatureExtractor(img_size=img_size)

    full_data_dir = os.path.join(data_dir, mode)
    class_names = list(split_info.keys()) if split_info else sorted(os.listdir(full_data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    all_samples = []
    for cls in class_names:
        cls_dir = os.path.join(full_data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"类别目录 {cls_dir} 不存在，跳过")
            continue
        img_files = [f for f in sorted(os.listdir(cls_dir))
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        for f in img_files:
            all_samples.append((os.path.join(cls_dir, f), class_to_idx[cls]))

    X_list, y_list = [], []
    failed_samples = []
    for img_path, label in tqdm(all_samples, desc=f"{mode} 特征提取"):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB').resize((img_size, img_size))
                X_list.append(feature_extractor.extract_all_features(np.array(img)))
                y_list.append(label)
        except:
            failed_samples.append(img_path)

    X = np.array(X_list)
    y = np.array(y_list)

    sample_weights = None
    if class_weights is not None:
        sample_weights = np.array([class_weights[class_names[label]] for label in y], dtype=np.float32)

    if failed_samples:
        print(f"{len(failed_samples)} 张图像处理失败")

    return X, y, sample_weights, class_names


def create_dataloaders(data_dir, img_size=224, batch_size=32, split_metadata=None):
    split_info = split_metadata.get('split_info', None) if split_metadata else None
    class_weights = split_metadata.get('class_weights', None) if split_metadata else None

    # 统一创建特征提取器
    feature_extractor = EnhancedFeatureExtractor(img_size=img_size)

    X_train, y_train, w_train, class_names = load_and_preprocess_data(
        data_dir, 'train', img_size, class_weights, split_info, feature_extractor
    )
    X_val, y_val, w_val, _ = load_and_preprocess_data(
        data_dir, 'val', img_size, class_weights, split_info, feature_extractor
    )
    X_test, y_test, w_test, _ = load_and_preprocess_data(
        data_dir, 'test', img_size, class_weights, split_info, feature_extractor
    )

    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    train_loader = DataLoaderSimple(X_train, y_train, w_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoaderSimple(X_val, y_val, w_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoaderSimple(X_test, y_test, w_test, batch_size=batch_size, shuffle=False)

    feature_dim = feature_extractor.get_feature_dimension()

    sample_weights_info = {'train_weights': w_train, 'val_weights': w_val, 'test_weights': w_test}

    return train_loader, val_loader, test_loader, class_names, feature_dim, class_to_idx, sample_weights_info

