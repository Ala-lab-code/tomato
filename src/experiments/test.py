import json
import os
import sys
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

# -----------------------------
# 基本路径设置
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/CNN")
TEST_DIR = os.path.join(BASE_DIR, "data/processed/test")
os.makedirs(CKPT_DIR, exist_ok=True)
# 读取 split_metadata.json
with open(os.path.join(PROCESSED_DATA_DIR, "split_metadata.json"), "r") as f:
    split_metadata = json.load(f)

# -----------------------------
# 设备设置
# -----------------------------
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# -----------------------------
# 数据集
# -----------------------------
test_dataset = TomatoDataset(TEST_DIR, mode="val")  # 测试集
test_loader = DataLoader(test_dataset, batch_size=16)

# -----------------------------
# 模型
# -----------------------------
model = ResNet50_SE(num_classes=10, pretrained=False)  # 测试不用加载预训练
model_path = os.path.join(CKPT_DIR, "best_model.ckpt")  # 替换为你训练的模型

# -----------------------------
# 加载训练好的权重
# -----------------------------
state_dict = paddle.load(model_path)
model.set_state_dict(state_dict)
model.eval()

# -----------------------------
# 损失函数
# -----------------------------
class_weights = split_metadata.get('class_weights')
class_weights_tensor = paddle.to_tensor(class_weights, dtype='float32')
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor) # 设置权重

# -----------------------------
# Runner
# -----------------------------
runner = Runner(model, optimizer=None, loss_fn=loss_fn, device=device)

# -----------------------------
# 测试
# -----------------------------
acc, avg_loss = runner.evaluate_loader(test_loader)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}")
