import sys
import os
import pickle
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader

# -----------------------------
# 路径设置（保证可 import src）
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner

# -----------------------------
# 设备设置
# -----------------------------
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# -----------------------------
# 基本路径
# -----------------------------
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------
# 数据集
# -----------------------------
train_dir = os.path.join(BASE_DIR, "data/processed/train")
val_dir = os.path.join(BASE_DIR, "data/processed/val")

train_dataset = TomatoDataset(train_dir, mode="train")
val_dataset = TomatoDataset(val_dir, mode="val")

# -----------------------------
# 超参数搜索空间（按你的要求）
# -----------------------------
learning_rates = [1e-3, 3e-4, 1e-4]
dropout_rates = [0.3, 0.5, 0.7]
batch_size = 16

results = []

for lr in learning_rates:
    for dropout in dropout_rates:
        print(f"\n=== Training lr={lr}, dropout={dropout} ===")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 模型
        model = ResNet50_SE(num_classes=10, pretrained=True, dropout_rate=dropout)

        # 冻结 backbone，解冻 layer4
        for param in model.backbone.parameters():
            param.stop_gradient = True
        for param in model.backbone[7].parameters():
            param.stop_gradient = False

        # 损失函数和优化器
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr)

        # Runner
        runner = Runner(model, optimizer, loss_fn)

        # 训练
        runner.train(
            train_loader,
            val_loader,
            num_epochs=10,
            save_path=f"{CKPT_DIR}/best_lr{lr}_dropout{dropout}.pdparams"
        )

        # 保存结果
        results.append({
            "lr": lr,
            "dropout": dropout,
            "train_loss": runner.train_epoch_losses,
            "train_acc": runner.train_epoch_accs,
            "val_loss": runner.val_epoch_losses,
            "val_acc": runner.val_epoch_accs,
        })

# -----------------------------
# 保存超参数搜索结果
# -----------------------------
result_path = os.path.join(CKPT_DIR, "hyperparam_results.pkl")
with open(result_path, "wb") as f:
    pickle.dump(results, f)

print(f"\nAll results saved to: {result_path}")
