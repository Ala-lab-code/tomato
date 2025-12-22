import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class TomatoLeafTransform:
    """番茄叶片病害数据增强变换"""

    @staticmethod
    def get_basic_transform(img_size=224):
        """
        基础预处理变换（验证集和测试集使用）

        步骤：
        1. Resize到256x256
        2. 中心裁剪到224x224
        3. 转换为Tensor（自动将像素值从[0,255]缩放到[0,1]）
        4. 标准化到[-1, 1]范围
        """
        return transforms.Compose([
            transforms.Resize(256),  # 先调整到256
            transforms.CenterCrop(img_size),  # 中心裁剪到指定大小
            transforms.ToTensor(),  # 转换为张量并自动缩放[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化到[-1, 1]
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_standard_augmentation(img_size=224):
        """
        标准数据增强（训练集使用）
        针对中等规模数据集（11607张图片，10个类别）

        增强步骤：
        1. Resize到256x256
        2. 随机裁剪并调整到224x224
        3. 随机水平翻转（50%概率）
        4. 随机垂直翻转（20%概率）
        5. 随机旋转（±15度）
        6. 颜色抖动（亮度、对比度、饱和度、色调）
        7. 随机仿射变换（平移10%）
        8. 转换为Tensor
        9. 标准化到[-1, 1]范围
        """
        return transforms.Compose([
            transforms.Resize(256),  # 先调整到256
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2,  # 颜色抖动
                                   saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
            transforms.ToTensor(),  # 转换为张量并自动缩放[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化到[-1, 1]
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_advanced_augmentation(img_size=224):
        """
        高级数据增强（训练集使用）
        针对叶片数据的优化增强

        增强步骤：
        1. Resize到256x256
        2. 随机裁剪并调整到224x224（更大的裁剪范围）
        3. 随机水平翻转（50%概率）
        4. 随机垂直翻转（30%概率，叶片可能上下对称）
        5. 随机旋转（±30度，叶片可能任意角度）
        6. 更强的颜色抖动
        7. 随机仿射变换（平移15%）
        8. 随机灰度化（10%概率）
        9. 高斯模糊（10%概率）
        10. 转换为Tensor
        11. 标准化到[-1, 1]范围
        """
        return transforms.Compose([
            transforms.Resize(256),  # 先调整到256
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),  # 更大的裁剪范围
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.3),  # 随机垂直翻转（叶片可能上下对称）
            transforms.RandomRotation(30),  # 更大的旋转角度（叶片可能任意角度）
            transforms.ColorJitter(brightness=0.3, contrast=0.3,  # 更强的颜色抖动
                                   saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # 更大的平移
            transforms.RandomGrayscale(p=0.1),  # 10%概率转为灰度
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊
            transforms.ToTensor(),  # 转换为张量并自动缩放[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化到[-1, 1]
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_light_augmentation(img_size=224):
        """
        轻量级数据增强（适用于小数据集或快速实验）

        增强步骤：
        1. Resize到256x256
        2. 随机裁剪并调整到224x224
        3. 随机水平翻转（30%概率）
        4. 随机旋转（±10度）
        5. 转换为Tensor
        6. 标准化到[-1, 1]范围
        """
        return transforms.Compose([
            transforms.Resize(256),  # 先调整到256
            transforms.RandomResizedCrop(img_size),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转
            transforms.ToTensor(),  # 转换为张量并自动缩放[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化到[-1, 1]
                                 std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_albumentations_augmentation(img_size=224):
        """
        使用Albumentations库的数据增强

        Albumentations优势：
        1. 更快的增强速度
        2. 更丰富的增强操作
        3. 支持更多图像格式
        """
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    @staticmethod
    def get_recommended_augmentation(img_size=224, dataset_size=11607, num_classes=10):
        """
        根据数据集规模和类别数量获取推荐的数据增强

        Args:
            img_size: 图像大小
            dataset_size: 数据集大小
            num_classes: 类别数量

        Returns:
            推荐的数据增强
        """
        # 计算每个类别的平均样本数
        avg_samples_per_class = dataset_size // num_classes

        print(f"数据集信息:")
        print(f"  - 总图片数: {dataset_size}")
        print(f"  - 类别数: {num_classes}")
        print(f"  - 平均每类: {avg_samples_per_class} 张")

        # 根据数据集规模推荐增强策略
        if dataset_size < 5000:
            print(f"推荐增强: 轻量级增强")
            return TomatoLeafTransform.get_light_augmentation(img_size)
        elif 5000 <= dataset_size <= 20000:
            if avg_samples_per_class >= 1000:
                print(f"推荐增强: 标准增强（每个类别样本充足）")
                return TomatoLeafTransform.get_standard_augmentation(img_size)
            else:
                print(f"推荐增强: 高级增强（需要更多数据多样性）")
                return TomatoLeafTransform.get_advanced_augmentation(img_size)
        else:
            print(f"推荐增强: 标准增强（大数据集）")
            return TomatoLeafTransform.get_standard_augmentation(img_size)

    @staticmethod
    def get_transform(mode='train', augmentation_type='recommended', img_size=224):
        """
        根据模式和类型获取变换

        Args:
            mode: 'train'（训练集）, 'val'（验证集）, 'test'（测试集）
            augmentation_type: 'basic', 'standard', 'advanced', 'light', 'albumentations', 'recommended'
            img_size: 图像大小

        Returns:
            相应的变换
        """
        if mode == 'train':
            if augmentation_type == 'basic':
                return TomatoLeafTransform.get_basic_transform(img_size)
            elif augmentation_type == 'standard':
                return TomatoLeafTransform.get_standard_augmentation(img_size)
            elif augmentation_type == 'advanced':
                return TomatoLeafTransform.get_advanced_augmentation(img_size)
            elif augmentation_type == 'light':
                return TomatoLeafTransform.get_light_augmentation(img_size)
            elif augmentation_type == 'albumentations':
                return TomatoLeafTransform.get_albumentations_augmentation(img_size)
            elif augmentation_type == 'recommended':
                return TomatoLeafTransform.get_recommended_augmentation(img_size)
            else:
                return TomatoLeafTransform.get_standard_augmentation(img_size)
        else:  # val 和 test 模式
            return TomatoLeafTransform.get_basic_transform(img_size)

    @staticmethod
    def get_inverse_transform():
        """
        获取逆变换（用于可视化处理后的图像）
        将标准化后的图像还原到[0,1]范围
        """
        return transforms.Compose([
            transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        ])

    @staticmethod
    def get_statistics():
        """
        获取标准化使用的统计信息
        """
        return {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'normalized_range': '[-1, 1]',
            'original_range': '[0, 1]'
        }


class AlbumentationsWrapper:
    """Albumentations变换包装器，用于兼容torchvision接口"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        image_np = np.array(image)
        augmented = self.transform(image=image_np)
        return augmented['image']


# 预定义的变换组合（方便直接导入使用）
BASIC_TRANSFORM = TomatoLeafTransform.get_basic_transform(224)
STANDARD_AUGMENTATION = TomatoLeafTransform.get_standard_augmentation(224)
ADVANCED_AUGMENTATION = TomatoLeafTransform.get_advanced_augmentation(224)
LIGHT_AUGMENTATION = TomatoLeafTransform.get_light_augmentation(224)
RECOMMENDED_AUGMENTATION = TomatoLeafTransform.get_recommended_augmentation(224)

