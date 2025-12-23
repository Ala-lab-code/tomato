import paddle
from paddle.io import DataLoader
from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner
import os

BASE_DIR = "/content/drive/MyDrive/tomato"
CKPT_DIR = f"{BASE_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# 数据集
train_dataset = TomatoDataset("data/processed/train", mode='train')
val_dataset = TomatoDataset("data/processed/val", mode='val')
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 模型
model = ResNet50_SE(num_classes=10, pretrained=True)

# 冻结 backbone 层
for param in model.backbone.parameters():
    param.stop_gradient = True
for param in model.backbone[7].parameters():  # 解冻 layer4
    param.stop_gradient = False

# 损失函数和优化器
loss_fn = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-4)

# Runner
runner = Runner(model, optimizer, loss_fn)

# 训练
runner.train(
    train_loader,
    val_loader,
    num_epochs=10,
    save_path=f"{CKPT_DIR}/best_resnet_se.pdparams",
    patience=5
)
