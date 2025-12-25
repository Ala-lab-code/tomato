import sys
import os
import json
import pickle
import shutil
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
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
CKPT_ROOT = os.path.join(BASE_DIR, "checkpoints/CNN")
os.makedirs(CKPT_ROOT, exist_ok=True)

# --------------------------------------------------
# 读取 split_metadata.json（类别权重）
# --------------------------------------------------
with open(os.path.join(PROCESSED_DATA_DIR, "split_metadata.json"), "r") as f:
    split_metadata = json.load(f)

class_weights = split_metadata["class_weights"]
class_weights_tensor = paddle.to_tensor(class_weights, dtype="float32")

# --------------------------------------------------
# 设备
# --------------------------------------------------
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# --------------------------------------------------
# 数据集
# --------------------------------------------------
train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
val_dir = os.path.join(PROCESSED_DATA_DIR, "val")

train_dataset = TomatoDataset(train_dir, mode="train")
val_dataset = TomatoDataset(val_dir, mode="val")

# --------------------------------------------------
# 超参数搜索空间
# --------------------------------------------------
learning_rates = [1e-3, 3e-4, 1e-4]
dropout_rates = [0.3, 0.5, 0.7]
batch_size = 16
num_epochs = 6
patience = 3

results = []

# --------------------------------------------------
# 超参数搜索
# --------------------------------------------------
for lr in learning_rates:
    for dropout in dropout_rates:
        print(f"\n=== Training lr={lr}, dropout={dropout} ===")

        exp_name = f"lr{lr}_dropout{dropout}"
        exp_ckpt_dir = os.path.join(CKPT_ROOT, exp_name)
        os.makedirs(exp_ckpt_dir, exist_ok=True)
        last_ckpt_path = os.path.join(exp_ckpt_dir, "last.ckpt")

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # --------------------------------------------------
        # 模型
        # --------------------------------------------------
        model = ResNet50_SE(num_classes=10, pretrained=True, dropout_rate=dropout)

        # 冻结 backbone，仅训练 layer4
        for param in model.backbone.parameters():
            param.stop_gradient = True
        for param in model.backbone[7].parameters():
            param.stop_gradient = False

        # --------------------------------------------------
        # Loss + Optimizer（类别权重只在 loss 中起作用）
        # --------------------------------------------------
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr)

        runner = Runner(model, optimizer, loss_fn, device=device)

        # --------------------------------------------------
        # Resume（如存在）
        # --------------------------------------------------
        if os.path.exists(last_ckpt_path):
            print(f"Resuming from {last_ckpt_path}")
            start_epoch = runner.load_checkpoint(last_ckpt_path)
        else:
            start_epoch = 0

        # --------------------------------------------------
        # 训练
        # --------------------------------------------------
        runner.train(
            train_loader=train_loader,
            dev_loader=val_loader,
            num_epochs=num_epochs,
            start_epoch=start_epoch,
            patience=patience,
            save_dir=exp_ckpt_dir
        )

        # --------------------------------------------------
        # 保存结果（用于画图 & 对比）
        # --------------------------------------------------
        results.append({
            "lr": lr,
            "dropout": dropout,
            "train_loss": runner.train_epoch_losses,
            "train_acc": runner.train_epoch_accs,
            "train_f1": runner.train_epoch_f1,
            "val_loss": runner.val_epoch_losses,
            "val_acc": runner.val_epoch_accs,
            "val_f1": runner.val_epoch_f1,
            "ckpt_dir": exp_ckpt_dir
        })

# --------------------------------------------------
# 保存所有实验结果
# --------------------------------------------------
result_path = os.path.join(CKPT_ROOT, "hyperparam_results.pkl")
with open(result_path, "wb") as f:
    pickle.dump(results, f)

print(f"\nAll results saved to: {result_path}")

# --------------------------------------------------
# 选最优模型（Val Acc）
# --------------------------------------------------
best_exp = max(results, key=lambda x: max(x["val_acc"]))
best_lr = best_exp["lr"]
best_dropout = best_exp["dropout"]
best_val_acc = max(best_exp["val_acc"])

best_ckpt_dir = best_exp["ckpt_dir"]
best_ckpt_path = os.path.join(best_ckpt_dir, "best.ckpt")

final_model_path = os.path.join(CKPT_ROOT, "best_model.ckpt")
shutil.copy(best_ckpt_path, final_model_path)

print("\n===== Best CNN Model =====")
print(f"lr={best_lr}, dropout={best_dropout}")
print(f"Best Val Acc={best_val_acc:.4f}")
print(f"Saved to: {final_model_path}")
