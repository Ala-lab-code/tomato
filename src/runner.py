# runner.py
import os
import paddle
from paddle.io import DataLoader
from tqdm import tqdm


class Runner:
    """
    Paddle 迁移学习训练 Runner（支持断点续训）
    功能：
    - 训练 / 验证日志
    - 保存 best model + last checkpoint
    - Early Stopping
    - 支持 resume 训练（Colab 断线可继续）
    """

    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # 训练过程记录
        self.train_epoch_losses = []
        self.train_epoch_accs = []
        self.val_epoch_losses = []
        self.val_epoch_accs = []

        # 最优模型
        self.best_score = 0.0
        self.best_epoch = -1

        # 设置设备
        if device is None:
            self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        else:
            self.device = device
        paddle.set_device(self.device)

    # ======================================================
    # 训练主函数
    # ======================================================
    def train(
        self,
        train_loader: DataLoader,
        dev_loader: DataLoader = None,
        num_epochs=20,
        start_epoch=0,
        patience=5,
        save_dir="checkpoints",
        compute_train_metrics=True,
    ):
        """
        start_epoch > 0 时即为 resume 训练
        """
        os.makedirs(save_dir, exist_ok=True)

        no_improve_epochs = 0
        self.model.train()

        for epoch in range(start_epoch, num_epochs):
            total_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                ncols=100,
            )

            for imgs, labels in pbar:
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ===== 训练集指标 =====
            if compute_train_metrics:
                train_acc, train_loss = self.evaluate_loader(train_loader)
            else:
                train_acc, train_loss = 0.0, total_loss / len(train_loader)

            # ===== 验证集指标 =====
            if dev_loader:
                val_acc, val_loss = self.evaluate_loader(dev_loader)
            else:
                val_acc, val_loss = 0.0, 0.0

            self.train_epoch_losses.append(train_loss)
            self.train_epoch_accs.append(train_acc)
            self.val_epoch_losses.append(val_loss)
            self.val_epoch_accs.append(val_acc)

            print(
                f"[Epoch {epoch + 1}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # ===== 保存 last checkpoint（每个 epoch）=====
            self.save_checkpoint(
                os.path.join(save_dir, "last.ckpt"),
                epoch
            )

            # ===== Early Stopping & best model =====
            if dev_loader:
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_epoch = epoch
                    no_improve_epochs = 0

                    self.save_checkpoint(
                        os.path.join(save_dir, "best.ckpt"),
                        epoch
                    )
                    print(f"Saved best model (Val Acc={val_acc:.4f})")

                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print(
                        f"⏹ Early stopping at epoch {epoch + 1}. "
                        f"Best Val Acc: {self.best_score:.4f} "
                        f"(epoch {self.best_epoch + 1})"
                    )
                    break

        print(
            f"Training finished. Best Val Acc: {self.best_score:.4f} "
            f"at epoch {self.best_epoch + 1}"
        )

    # ======================================================
    # 验证 / 推理
    # ======================================================
    @paddle.no_grad()
    def evaluate_loader(self, loader: DataLoader):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels).item()
            total_loss += loss

            preds = paddle.argmax(logits, axis=1)
            correct += (preds == labels).astype("float32").sum().item()
            total += labels.shape[0]

        self.model.train()
        return total_loss / len(loader), correct / total if total > 0 else 0.0

    @paddle.no_grad()
    def predict(self, imgs):
        self.model.eval()
        return self.model(imgs)

    # ======================================================
    # 保存 / 加载 checkpoint
    # ======================================================
    def save_checkpoint(self, path, epoch):
        paddle.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "best_score": self.best_score,
                "train_losses": self.train_epoch_losses,
                "val_losses": self.val_epoch_losses,
            },
            path,
        )

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        ckpt = paddle.load(path)

        self.model.set_state_dict(ckpt["model"])
        self.optimizer.set_state_dict(ckpt["optimizer"])
        self.best_score = ckpt.get("best_score", 0.0)

        self.train_epoch_losses = ckpt.get("train_losses", [])
        self.val_epoch_losses = ckpt.get("val_losses", [])

        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resume training from epoch {start_epoch + 1}")
        return start_epoch
