#dataset.py
import os
from PIL import Image
from paddle.io import Dataset
from paddle.vision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize


class TomatoDataset(Dataset):
    """TomatoDataset
    自定义 Tomato 叶片病害图像数据集
    - 支持 train / val / test
    - train 集合支持在线增强
    - val / test 集合只做 Resize + Normalize
    """

    def __init__(self, data_dir, mode='train'):
        """
        Args:
            data_dir (str): 数据集路径，例如 data/processed/train
            mode (str): 'train' / 'val' / 'test'
        """
        super(TomatoDataset, self).__init__()
        self.mode = mode
        self.data_dir = data_dir

        # 遍历类别
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集图片路径和标签
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        # 定义 transform
        if mode == 'train':
            self.transform = Compose([
                Resize((256, 256)),                # 先统一尺寸
                RandomCrop((224, 224)),           # 随机裁剪
                RandomHorizontalFlip(),            # 随机水平翻转
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色扰动
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:  # val / test
            self.transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)


