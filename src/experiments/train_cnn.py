# train_cnn.py
import sys
import os
import pickle
import shutil
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
import json
from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner

# -----------------------------
# 路径设置
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(BASE_DIR)

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data/processed")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints/CNN")
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------
# 读取 split_metadata.json
# -----------------------------
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

# -----------------------------
# 设备设置
# -----------------------------
device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
paddle.set_device(device)

# -----------------------------
# 数据集
# -----------------------------
train_dir = os.path.join(BASE_DIR, "data/processed/train")
val_dir = os.path.join(BASE_DIR, "data/processed/val")

train_dataset = TomatoDataset(train_dir, mode="train")
val_dataset = TomatoDataset(val_dir, mode="val")

# -----------------------------
# 超参数搜索空间
# -----------------------------
learning_rates = [1e-3,3e-3,1e-4,]
dropout_rates = [0.3, 0.5, 0.7]
batch_size = 16

# -----------------------------
# 加载已有结果
# -----------------------------
result_path = os.path.join(CKPT_DIR, "hyperparam_results.pkl")
if os.path.exists(result_path):
    with open(result_path, "rb") as f:
        results = pickle.load(f)
    print(f"已加载 {len(results)} 个组合的历史结果，继续训练未完成组合")
else:
    results = []

# -----------------------------
# 超参数搜索循环
# -----------------------------
for lr in learning_rates:
    for dropout in dropout_rates:
        # 检查是否已训练过
        if any(r['lr'] == lr and r['dropout'] == dropout for r in results):
            print(f"组合 lr={lr}, dropout={dropout} 已训练过，跳过")
            continue

        print(f"\n=== Training lr={lr}, dropout={dropout} ===")

        # 每个实验独立目录
        exp_name = f"lr{lr}_dropout{dropout}"
        exp_ckpt_dir = os.path.join(CKPT_DIR, exp_name)
        os.makedirs(exp_ckpt_dir, exist_ok=True)
        last_ckpt_path = os.path.join(exp_ckpt_dir, "last.ckpt")

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 模型
        model = ResNet50_SE(num_classes=10, pretrained=True, dropout_rate=dropout)

        # 冻结 backbone，仅训练 layer4
        for param in model.backbone.parameters():
            param.stop_gradient = True
        for param in model.backbone[7].parameters():
            param.stop_gradient = False

        # 损失函数 & 优化器
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr)

        runner = Runner(model, optimizer, loss_fn, device=device)

        # 自动 resume
        if os.path.exists(last_ckpt_path):
            print(f"Resuming from {last_ckpt_path}")
            start_epoch = runner.load_checkpoint(last_ckpt_path)
        else:
            print("Starting from scratch")
            start_epoch = 0

        # 训练
        runner.train(
            train_loader,
            val_loader,
            num_epochs=6,
            start_epoch=start_epoch,
            patience=3,
            save_dir=exp_ckpt_dir
        )

        # 记录结果
        results.append({
            "lr": lr,
            "dropout": dropout,
            "train_loss": runner.train_epoch_losses,
            "train_acc": runner.train_epoch_accs,
            "train_f1": runner.train_epoch_f1,
            "train_precision": runner.train_epoch_precision,
            "train_recall": runner.train_epoch_recall,
            "val_loss": runner.val_epoch_losses,
            "val_acc": runner.val_epoch_accs,
            "val_f1": runner.val_epoch_f1,
            "val_precision": runner.val_epoch_precision,
            "val_recall": runner.val_epoch_recall,
            "ckpt_dir": exp_ckpt_dir
        })

        # 实时保存 results
        with open(result_path, "wb") as f:
            pickle.dump(results, f)
        print(f"已保存 hyperparam_results.pkl (共 {len(results)} 个组合)")

# -----------------------------
# 选出最佳组合并保存模型
# -----------------------------
best_exp = max(results, key=lambda x: max(x["val_acc"]))
best_lr = best_exp["lr"]
best_dropout = best_exp["dropout"]
best_val_acc = max(best_exp["val_acc"])
best_ckpt_dir = best_exp["ckpt_dir"]
best_ckpt_path = os.path.join(best_ckpt_dir, "best.ckpt")

final_model_path = os.path.join(CKPT_DIR, "best_model.ckpt")
shutil.copy(best_ckpt_path, final_model_path)

print(f"\n最佳超参数组合: lr={best_lr}, dropout={best_dropout}")
print(f"最佳验证集准确率: {best_val_acc:.4f}")
print(f"最佳模型已保存到: {final_model_path}")
