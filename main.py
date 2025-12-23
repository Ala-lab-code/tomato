import paddle
print("CUDA:", paddle.is_compiled_with_cuda())
paddle.set_device('gpu')

import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.metric import Accuracy
from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner
import os

BASE_DIR = "/content/drive/MyDrive/tomato"
CKPT_DIR = f"{BASE_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------
# 数据集
# -----------------------------
train_dataset = TomatoDataset("data/processed/train", mode='train')
val_dataset = TomatoDataset("data/processed/val", mode='val')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -----------------------------
# 模型
# -----------------------------
model = ResNet50_SE(num_classes=10, pretrained=True)

# -----------------------------
# 冻结 ResNet50 前面层，只训练 layer4 + SE + 分类头
# -----------------------------
# 先冻结全部 backbone
for param in model.backbone.parameters():
    param.stop_gradient = True

# 解冻 layer4
for param in model.backbone[7].parameters():
    param.stop_gradient = False

# -----------------------------
# 损失函数、优化器和指标
# -----------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=1e-4)
metric = Accuracy()

# -----------------------------
# Runner
# -----------------------------
runner = Runner(model, optimizer, loss_fn, metric)

runner.train(
    train_loader,
    val_loader,
    num_epochs=10,
    log_steps=50,
    eval_steps=200,
    save_path=f"{CKPT_DIR}/best_resnet_se.pdparams"
)
