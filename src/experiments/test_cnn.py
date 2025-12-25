# test_cnn.py
import os
import sys
import json
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

# --------------------------------------------------
# 路径设置
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/CNN")
TEST_DIR = os.path.join(PROCESSED_DATA_DIR, "test")

# --------------------------------------------------
# 读取类别权重
# --------------------------------------------------
with open(os.path.join(PROCESSED_DATA_DIR, "split_metadata.json"), "r") as f:
    split_metadata = json.load(f)

class_order = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

class_weights = [split_metadata["class_weights"][k] for k in class_order]
class_weights_tensor = paddle.to_tensor(class_weights, dtype="float32")


# --------------------------------------------------
# 设备
# --------------------------------------------------
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# --------------------------------------------------
# 测试数据
# --------------------------------------------------
test_dataset = TomatoDataset(TEST_DIR, mode="val")
test_loader = DataLoader(test_dataset, batch_size=16)

# --------------------------------------------------
# 模型
# --------------------------------------------------
model = ResNet50_SE(num_classes=10, pretrained=False)

ckpt_path = os.path.join(CKPT_DIR, "best_model.ckpt")
ckpt = paddle.load(ckpt_path)
model.set_state_dict(ckpt["model"])
model.eval()

# --------------------------------------------------
# 损失函数（带类别权重）
# --------------------------------------------------
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

# --------------------------------------------------
# Runner（测试不需要 optimizer）
# --------------------------------------------------
runner = Runner(
    model=model,
    optimizer=None,
    loss_fn=loss_fn,
    device=device
)

# --------------------------------------------------
# 测试
# --------------------------------------------------
test_loss, test_acc, test_f1, test_p, test_r = runner.evaluate_loader(test_loader)

print("\n===== Test Results (CNN) =====")
print(f"Loss       : {test_loss:.4f}")
print(f"Accuracy   : {test_acc:.4f}")
print(f"F1-weighted: {test_f1:.4f}")
print(f"Precision  : {test_p:.4f}")
print(f"Recall     : {test_r:.4f}")
