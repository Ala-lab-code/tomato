"""
data_processor.py
数据集加载、预处理和增强
将dataset和preprocess功能合并到一个文件中
"""
"""data_processor.py
数据集加载、预处理和增强
将dataset和preprocess功能合并到一个文件中
"""

import os
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

# 修改导入路径，确保能正确导入transforms.py中的类
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 尝试从当前目录导入
    from transforms import TomatoLeafTransform
except ImportError:
    # 如果当前目录没有，尝试从父目录导入
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.data.transforms import TomatoLeafTransform
    except ImportError:
        # 如果都失败，使用简化的版本
        import torchvision.transforms as transforms


        class TomatoLeafTransform:
            @staticmethod
            def get_basic_transform(img_size=224):
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

            @staticmethod
            def get_standard_augmentation(img_size=224):
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                           saturation=0.2, hue=0.1),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

            # 添加缺少的get_transform方法
            @staticmethod
            def get_transform(mode='train', augmentation_type='recommended', img_size=224):
                if mode == 'train':
                    if augmentation_type == 'basic':
                        return TomatoLeafTransform.get_basic_transform(img_size)
                    elif augmentation_type == 'standard':
                        return TomatoLeafTransform.get_standard_augmentation(img_size)
                    else:
                        return TomatoLeafTransform.get_standard_augmentation(img_size)
                else:  # val 和 test 模式
                    return TomatoLeafTransform.get_basic_transform(img_size)

class TomatoDataset(Dataset):
    """番茄病害数据集加载类"""

    def __init__(self, data_dir, transform=None, mode='train', class_subset=None):
        """
        初始化数据集

        Args:
            data_dir: 数据根目录（包含train/val/test子目录）
            transform: 数据变换
            mode: 数据集模式 ('train', 'val', 'test')
            class_subset: 可选的类别子集列表
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        self.image_paths = []
        self.labels = []

        # 构建完整的数据路径
        full_data_dir = os.path.join(data_dir, mode)

        if not os.path.exists(full_data_dir):
            raise ValueError(f"数据集路径不存在: {full_data_dir}")

        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(full_data_dir)
                               if os.path.isdir(os.path.join(full_data_dir, d))])

        # 如果指定了类别子集，则过滤类别
        if class_subset is not None:
            self.classes = [cls for cls in self.classes if cls in class_subset]
            if len(self.classes) == 0:
                raise ValueError(f"类别子集 {class_subset} 在数据集中未找到任何匹配的类别")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集所有样本
        for class_name in self.classes:
            class_dir = os.path.join(full_data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in sorted(os.listdir(class_dir)):
                if self._is_image_file(img_name):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        if len(self.samples) == 0:
            raise ValueError(f"在 {full_data_dir} 中没有找到任何图像文件!")

        print(f"加载 {len(self.samples)} 张图片 for {mode} 数据集")
        print(f"类别 ({len(self.classes)}): {self.classes}")

        # 打印类别分布
        self._print_class_distribution()

    def _is_image_file(self, filename):
        """检查文件是否为图像文件"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp')
        return filename.lower().endswith(image_extensions)

    def _print_class_distribution(self):
        """打印类别分布"""
        print(f"\n{self.mode} 数据集类别分布:")
        print("-" * 50)
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for _, label in self.samples if label == class_idx)
            print(f"  {class_name}: {count} 张图片")
        print("-" * 50)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # 打开图像并转换为RGB
            image = Image.open(img_path).convert('RGB')

            # 应用变换
            if self.transform:
                image = self.transform(image)

        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            # 如果图像损坏，返回黑色图像
            if self.transform:
                # 如果是torchvision变换，返回零张量
                image = torch.zeros((3, 224, 224))
                if hasattr(self.transform, '__call__'):
                    image = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                image = Image.new('RGB', (224, 224), color='black')

        return image, label, img_path

    def get_class_names(self):
        """获取类别名称列表"""
        return self.classes

    def get_class_distribution(self):
        """获取类别分布字典"""
        distribution = {}
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for _, label in self.samples if label == class_idx)
            distribution[class_name] = count
        return distribution


class DataPreprocessor:
    """数据集预处理器（预处理和增强）"""

    @staticmethod
    def preprocess_and_save(input_dir, output_dir, img_size=224,
                            train_augmentation='standard',
                            val_test_augmentation='basic',
                            random_seed=42):
        """
        预处理图像并保存到新目录

        Args:
            input_dir: 输入目录（包含train/val/test子目录）
            output_dir: 输出目录
            img_size: 图像大小
            train_augmentation: 训练集增强类型
            val_test_augmentation: 验证/测试集增强类型
            random_seed: 随机种子

        Returns:
            统计信息字典
        """
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # 检查输入目录
        if not os.path.exists(input_dir):
            raise ValueError(f"输入目录不存在: {input_dir}")

        # 定义分割集
        splits = ['train', 'val', 'test']

        # 创建输出目录
        for split in splits:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)

        # 获取变换
        train_transform = TomatoLeafTransform.get_transform(
            'train', train_augmentation, img_size
        )

        val_test_transform = TomatoLeafTransform.get_transform(
            'val', val_test_augmentation, img_size
        )

        # 统计信息
        stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'test_images': 0,
            'class_names': [],
            'class_counts': {}
        }

        print(f"开始预处理数据集...")
        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")
        print(f"图像大小: {img_size}x{img_size}")
        print(f"训练集增强: {train_augmentation}")
        print(f"验证/测试集增强: {val_test_augmentation}")

        # 处理每个分割集
        for split in splits:
            print(f"\n处理{split}集...")

            split_input_dir = os.path.join(input_dir, split)
            split_output_dir = os.path.join(output_dir, split)

            if not os.path.exists(split_input_dir):
                print(f"警告: {split_input_dir} 不存在，跳过")
                continue

            # 获取类别
            classes = sorted([d for d in os.listdir(split_input_dir)
                              if os.path.isdir(os.path.join(split_input_dir, d))])

            if split == 'train':
                stats['class_names'] = classes

            for class_name in tqdm(classes, desc=f"处理{split}集类别"):
                class_input_dir = os.path.join(split_input_dir, class_name)
                class_output_dir = os.path.join(split_output_dir, class_name)
                os.makedirs(class_output_dir, exist_ok=True)

                # 获取图像文件
                image_files = [f for f in os.listdir(class_input_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))]

                if not image_files:
                    print(f"警告: {class_input_dir} 中没有图像文件")
                    continue

                # 更新统计信息
                if class_name not in stats['class_counts']:
                    stats['class_counts'][class_name] = {}
                stats['class_counts'][class_name][split] = len(image_files)

                if split == 'train':
                    stats['train_images'] += len(image_files)
                elif split == 'val':
                    stats['val_images'] += len(image_files)
                elif split == 'test':
                    stats['test_images'] += len(image_files)

                stats['total_images'] += len(image_files)

                # 处理每个图像
                for img_file in image_files:
                    img_path = os.path.join(class_input_dir, img_file)

                    try:
                        # 打开图像
                        img = Image.open(img_path).convert('RGB')

                        # 应用变换
                        if split == 'train':
                            transformed_img = train_transform(img)
                        else:
                            transformed_img = val_test_transform(img)

                        # 将处理后的图像保存为JPEG格式
                        # 首先需要将张量转换回PIL图像
                        if isinstance(transformed_img, torch.Tensor):
                            # 反标准化
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            img_np = transformed_img.cpu().numpy()
                            img_np = img_np * std.numpy() + mean.numpy()
                            img_np = np.clip(img_np, 0, 1)
                            img_np = (img_np * 255).astype(np.uint8)
                            img_np = np.transpose(img_np, (1, 2, 0))
                            save_img = Image.fromarray(img_np)
                        else:
                            save_img = transformed_img

                        # 保存图像
                        base_name = os.path.splitext(img_file)[0]
                        save_path = os.path.join(class_output_dir, f"{base_name}.jpg")
                        save_img.save(save_path, 'JPEG', quality=95)

                    except Exception as e:
                        print(f"处理图像 {img_path} 时出错: {e}")
                        # 如果出错，尝试直接复制
                        try:
                            shutil.copy2(img_path, os.path.join(class_output_dir, img_file))
                        except:
                            pass

        print(f"\n{'=' * 60}")
        print(f"预处理完成!")
        print(f"总图像数: {stats['total_images']}")
        print(f"训练集: {stats['train_images']} 张")
        print(f"验证集: {stats['val_images']} 张")
        print(f"测试集: {stats['test_images']} 张")
        print(f"类别数: {len(stats['class_names'])}")

        # 保存统计信息
        stats_file = os.path.join(output_dir, "preprocessing_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("数据集预处理统计信息\n")
            f.write("=" * 60 + "\n")
            f.write(f"输入目录: {input_dir}\n")
            f.write(f"输出目录: {output_dir}\n")
            f.write(f"图像大小: {img_size}x{img_size}\n")
            f.write(f"训练集增强: {train_augmentation}\n")
            f.write(f"验证/测试集增强: {val_test_augmentation}\n")
            f.write(f"随机种子: {random_seed}\n\n")

            f.write(f"总图像数: {stats['total_images']}\n")
            f.write(f"训练集: {stats['train_images']} 张\n")
            f.write(f"验证集: {stats['val_images']} 张\n")
            f.write(f"测试集: {stats['test_images']} 张\n")
            f.write(f"类别数: {len(stats['class_names'])}\n\n")

            f.write("各类别图像数量:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'类别':<40} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}\n")
            f.write("-" * 60 + "\n")

            for class_name in stats['class_names']:
                train_count = stats['class_counts'].get(class_name, {}).get('train', 0)
                val_count = stats['class_counts'].get(class_name, {}).get('val', 0)
                test_count = stats['class_counts'].get(class_name, {}).get('test', 0)
                total_count = train_count + val_count + test_count
                f.write(f"{class_name:<40} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}\n")

        print(f"\n统计信息已保存到: {stats_file}")
        print(f"处理后的数据集结构:")
        print(f"  {output_dir}/")
        print(f"    ├── train/")
        print(f"    ├── val/")
        print(f"    ├── test/")
        print(f"    └── preprocessing_stats.txt")

        return stats

    @staticmethod
    def create_dataloaders(data_dir, batch_size=32, num_workers=2,
                           transform_type='standard', img_size=224,
                           shuffle_train=True):
        """
        创建数据加载器

        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            num_workers: 工作线程数
            transform_type: 变换类型
            img_size: 图像大小
            shuffle_train: 是否打乱训练集

        Returns:
            train_loader, val_loader, test_loader, class_names
        """
        # 获取变换
        train_transform = TomatoLeafTransform.get_transform('train', transform_type, img_size)
        val_transform = TomatoLeafTransform.get_transform('val', 'basic', img_size)
        test_transform = TomatoLeafTransform.get_transform('test', 'basic', img_size)

        # 创建数据集
        train_dataset = TomatoDataset(
            data_dir=data_dir,
            transform=train_transform,
            mode='train'
        )

        val_dataset = TomatoDataset(
            data_dir=data_dir,
            transform=val_transform,
            mode='val'
        )

        test_dataset = TomatoDataset(
            data_dir=data_dir,
            transform=test_transform,
            mode='test'
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader, train_dataset.get_class_names()


def main():
    """主函数：执行数据集预处理"""

    print("=" * 70)
    print("番茄病害数据集预处理工具")
    print("=" * 70)
    print("功能:")
    print("  1. 加载已划分的数据集")
    print("  2. 应用数据增强和预处理")
    print("  3. 保存处理后的图像")
    print("  4. 创建数据加载器")
    print("=" * 70)

    # 配置参数
    INPUT_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\splits"  # 已划分的数据集目录
    OUTPUT_DIR = r"C:\Users\someb\Desktop\tomato_disease_classification\data\processed"  # 处理后的输出目录
    IMG_SIZE = 224  # 图像大小
    TRAIN_AUGMENTATION = 'standard'  # 训练集增强类型
    RANDOM_SEED = 42  # 随机种子

    print(f"\n配置信息:")
    print(f"  输入目录: {INPUT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  图像大小: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  训练集增强: {TRAIN_AUGMENTATION}")
    print(f"  随机种子: {RANDOM_SEED}")

    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"\n错误: 输入目录不存在: {INPUT_DIR}")
        print("请确保已运行数据集划分脚本")
        return

    # 检查输出目录是否已存在
    if os.path.exists(OUTPUT_DIR):
        print(f"\n警告: 输出目录 '{OUTPUT_DIR}' 已存在!")
        overwrite = input("是否覆盖? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("程序退出")
            return
        else:
            # 删除已存在的目录
            try:
                shutil.rmtree(OUTPUT_DIR)
                print(f"已删除旧目录: {OUTPUT_DIR}")
            except Exception as e:
                print(f"删除目录失败: {e}")
                return

    # 执行预处理
    try:
        stats = DataPreprocessor.preprocess_and_save(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            img_size=IMG_SIZE,
            train_augmentation=TRAIN_AUGMENTATION,
            val_test_augmentation='basic',
            random_seed=RANDOM_SEED
        )

        print("\n" + "=" * 70)
        print("数据集预处理完成!")
        print("=" * 70)

        # 显示结果
        print(f"\n处理结果:")
        print(f"  总图像数: {stats['total_images']}")
        print(f"  训练集: {stats['train_images']} 张 (已应用数据增强)")
        print(f"  验证集: {stats['val_images']} 张")
        print(f"  测试集: {stats['test_images']} 张")
        print(f"  类别数: {len(stats['class_names'])}")

        # 创建数据加载器测试
        print(f"\n创建数据加载器测试...")
        train_loader, val_loader, test_loader, class_names = DataPreprocessor.create_dataloaders(
            data_dir=OUTPUT_DIR,
            batch_size=4,  # 小批次用于测试
            num_workers=0,
            transform_type='standard',
            img_size=IMG_SIZE
        )

        print(f"数据加载器创建成功!")
        print(f"  训练集批次: {len(train_loader)}")
        print(f"  验证集批次: {len(val_loader)}")
        print(f"  测试集批次: {len(test_loader)}")
        print(f"  类别名称: {class_names}")

        # 显示一个批次的样本
        print(f"\n显示一个训练批次的样本...")
        for images, labels, paths in train_loader:
            print(f"  批次形状: {images.shape}")
            print(f"  标签: {labels}")
            break

    except Exception as e:
        print(f"\n预处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()