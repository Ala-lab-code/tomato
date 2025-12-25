# runner.py
import os
import paddle
from paddle.io import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


class Runner:
    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # 训练指标
        self.train_epoch_losses = []
        self.train_epoch_accs = []
        self.train_epoch_f1 = []
        self.train_epoch_precision = []
        self.train_epoch_recall = []

        # 验证指标
        self.val_epoch_losses = []
        self.val_epoch_accs = []
        self.val_epoch_f1 = []
        self.val_epoch_precision = []
        self.val_epoch_recall = []

        self.best_score = 0.0
        self.best_epoch = -1

        self.device = device or ("gpu" if paddle.is_compiled_with_cuda() else "cpu")
        paddle.set_device(self.device)

    # ======================================================
    # 训练
    # ======================================================
    def train(
        self,
        train_loader,
        dev_loader=None,
        num_epochs=10,
        start_epoch=0,
        patience=5,
        save_dir="checkpoints",
    ):
        os.makedirs(save_dir, exist_ok=True)
        no_improve_epochs = 0

        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for imgs, labels in pbar:
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ===== Epoch 级指标 =====
            train_loss, train_acc, train_f1, train_p, train_r = self.evaluate_loader(train_loader)

            if dev_loader:
                val_loss, val_acc, val_f1, val_p, val_r = self.evaluate_loader(dev_loader)
            else:
                val_loss = val_acc = val_f1 = val_p = val_r = 0.0

            # ===== 记录 =====
            self.train_epoch_losses.append(train_loss)
            self.train_epoch_accs.append(train_acc)
            self.train_epoch_f1.append(train_f1)
            self.train_epoch_precision.append(train_p)
            self.train_epoch_recall.append(train_r)

            self.val_epoch_losses.append(val_loss)
            self.val_epoch_accs.append(val_acc)
            self.val_epoch_f1.append(val_f1)
            self.val_epoch_precision.append(val_p)
            self.val_epoch_recall.append(val_r)

            print(
                f"[Epoch {epoch+1}] "
                f"Train Acc={train_acc:.4f}, F1={train_f1:.4f} | "
                f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}"
            )

            # ===== checkpoint =====
            self.save_checkpoint(os.path.join(save_dir, "last.ckpt"), epoch)

            if dev_loader:
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_epoch = epoch
                    no_improve_epochs = 0
                    self.save_checkpoint(os.path.join(save_dir, "best.ckpt"), epoch)
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    # ======================================================
    # 评估
    # ======================================================
    @paddle.no_grad()
    def evaluate_loader(self, loader):
        self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        for imgs, labels in loader:
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels)
            total_loss += loss.item()

            preds = paddle.argmax(logits, axis=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        return total_loss / len(loader), acc, f1, precision, recall

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
                "train_accs": self.train_epoch_accs,
                "train_f1": self.train_epoch_f1,
                "train_precision": self.train_epoch_precision,
                "train_recall": self.train_epoch_recall,
                "val_losses": self.val_epoch_losses,
                "val_accs": self.val_epoch_accs,
                "val_f1": self.val_epoch_f1,
                "val_precision": self.val_epoch_precision,
                "val_recall": self.val_epoch_recall,
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
        self.train_epoch_accs = ckpt.get("train_accs", [])
        self.train_epoch_f1 = ckpt.get("train_f1", [])
        self.train_epoch_precision = ckpt.get("train_precision", [])
        self.train_epoch_recall = ckpt.get("train_recall", [])

        self.val_epoch_losses = ckpt.get("val_losses", [])
        self.val_epoch_accs = ckpt.get("val_accs", [])
        self.val_epoch_f1 = ckpt.get("val_f1", [])
        self.val_epoch_precision = ckpt.get("val_precision", [])
        self.val_epoch_recall = ckpt.get("val_recall", [])

        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resume training from epoch {start_epoch + 1}")
        return start_epoch
