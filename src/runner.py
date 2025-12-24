import paddle
from paddle.io import DataLoader
from tqdm import tqdm

class Runner:
    """
    迁移学习训练封装类
    功能：
    - 支持 epoch 训练日志打印
    - 支持验证集评估
    - 保存验证集最优模型
    - 支持 Early Stopping
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

        # 最优模型相关
        self.best_score = 0.0
        self.best_epoch = 0

        # 设置设备
        if device is None:
            self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        else:
            self.device = device
        paddle.set_device(self.device)

    def train(self, train_loader: DataLoader, dev_loader: DataLoader = None,
              num_epochs=10, save_path="best_model.pdparams", patience=3,
              compute_train_metrics=True):
        """
        训练主函数
        """
        self.model.train()
        no_improve_epochs = 0  # Early Stopping计数

        for epoch in range(num_epochs):
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

            # 遍历 batch
            for imgs, labels in pbar:
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                total_loss += loss.item()
                pbar.update(1)

            # 一个 epoch 完成后统计指标
            if compute_train_metrics:
                train_acc, train_loss = self.evaluate_loader(train_loader)
            else:
                train_acc, train_loss = 0.0, total_loss / len(train_loader)

            if dev_loader:
                val_acc, val_loss = self.evaluate_loader(dev_loader)
            else:
                val_acc, val_loss = 0.0, 0.0

            # 保存指标到成员变量
            self.train_epoch_losses.append(train_loss)
            self.train_epoch_accs.append(train_acc)
            self.val_epoch_losses.append(val_loss)
            self.val_epoch_accs.append(val_acc)

            print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early Stopping 判断
            if dev_loader:
                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    print(f"Saved best model with Val Acc: {val_acc:.4f}")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best Val Acc: {self.best_score:.4f} at epoch {self.best_epoch + 1}"
                    )
                    break

        print(f"Training finished. Best Val Acc: {self.best_score:.4f} at epoch {self.best_epoch + 1}")

    @paddle.no_grad()
    def evaluate_loader(self, loader: DataLoader):
        """
        对任意 DataLoader 计算平均 loss 和 Accuracy（手动计算）
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in loader:
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels).item()
            total_loss += loss

            preds = paddle.argmax(logits, axis=1)
            correct += (preds == labels).astype('float32').sum().item()
            total += labels.shape[0]

        avg_loss = total_loss / len(loader)
        acc = correct / total if total > 0 else 0.0
        return acc, avg_loss

    @paddle.no_grad()
    def predict(self, imgs):
        self.model.eval()
        return self.model(imgs)

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        state_dict = paddle.load(model_path)
        self.model.set_state_dict(state_dict)
