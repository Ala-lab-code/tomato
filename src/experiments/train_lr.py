import os
import json
from src.features import create_dataloader
from src.models.baseline_lr import ImprovedLogisticRegressionCV

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CKPT_ROOT = os.path.join(BASE_DIR, "checkpoints/LR")
os.makedirs(CKPT_ROOT, exist_ok=True)
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")

# 读取 split_metadata.json
with open(os.path.join(PROCESSED_DATA_DIR, "split_metadata.json"), "r") as f:
    split_metadata = json.load(f)

# 创建 DataLoader
train_loader, feature_dim = create_dataloader(
    PROCESSED_DATA_DIR, split='train', img_size=224, batch_size=32,
    split_metadata=split_metadata
)
val_loader, _ = create_dataloader(
    PROCESSED_DATA_DIR, split='val', img_size=224, batch_size=32,
    split_metadata=split_metadata
)

# 初始化模型
model = ImprovedLogisticRegressionCV(num_classes=10, class_weights=split_metadata.get('class_weights'))

# 训练
history = model.train_with_validation(train_loader, val_loader, use_pca=True)

# 保存模型
model.save_model(os.path.join(CKPT_ROOT, "trained_lr_model.pkl"))

print("训练完成，模型已保存")
print("训练历史：", history)
