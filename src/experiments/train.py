import os
import pickle
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
from src.dataset import TomatoDataset
from src.models.resnet_se import ResNet50_SE
from src.runner import Runner
paddle.set_device('gpu')
# -----------------------------
# 基本路径设置
# -----------------------------
BASE_DIR = "/content/drive/MyDrive/tomato"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------------------
# 数据集
# -----------------------------
train_dataset = TomatoDataset("../../data/processed/train", mode='train')
val_dataset = TomatoDataset("../../data/processed/val", mode='val')

# -----------------------------
# 超参数搜索空间
# -----------------------------
learning_rates = [1e-3, 1e-4]
batch_sizes = [16, 32]
freeze_layer4_options = [True, False]  # 可选是否冻结 layer4

# 记录结果
results = []

# -----------------------------
# Grid Search
# -----------------------------
for lr in learning_rates:
    for batch_size in batch_sizes:
        for freeze_layer4 in freeze_layer4_options:
            print(f"\n=== Training with lr={lr}, batch_size={batch_size}, freeze_layer4={freeze_layer4} ===")

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # 模型
            model = ResNet50_SE(num_classes=10, pretrained=True)

            # 冻结 backbone 除最后一层
            for param in model.backbone.parameters():
                param.stop_gradient = True

            if not freeze_layer4:
                for param in model.backbone[7].parameters():  # layer4
                    param.stop_gradient = False

            # 损失函数和优化器
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(parameters=model.parameters(), learning_rate=lr)

            # Runner
            runner = Runner(model, optimizer, loss_fn)

            # 训练
            runner.train(train_loader, val_loader, num_epochs=5, patience=3,
                         save_path=os.path.join(
                             CKPT_DIR, f"best_lr{lr}_bs{batch_size}_freeze{freeze_layer4}.pdparams")
                         )

            # 保存结果
            results.append({
                "lr": lr,
                "batch_size": batch_size,
                "freeze_layer4": freeze_layer4,
                "train_loss": runner.train_epoch_losses,
                "train_acc": runner.train_epoch_accs,
                "val_loss": runner.val_epoch_losses,
                "val_acc": runner.val_epoch_accs
            })

# -----------------------------
# 保存结果到文件，下次可直接加载
# -----------------------------
with open(os.path.join(CKPT_DIR, "hyperparam_results.pkl"), "wb") as f:
    pickle.dump(results, f)
print("All results saved to hyperparam_results.pkl")
