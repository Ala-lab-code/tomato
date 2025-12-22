# src/runner.py
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.metric import Accuracy
from tqdm import tqdm  # 显示进度条

class Runner:
    """
    迁移学习训练封装
    支持 step/epoch 训练日志、验证、保存最优模型
    """
    def __init__(self, model, optimizer, loss_fn, metric=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric  # 用于计算评价指标，如 Accuracy

        # 记录训练过程
        self.train_step_losses = []
        self.train_epoch_losses = []
        self.dev_losses = []
        self.dev_scores = []

        self.best_score = 0.0  # 验证集最优指标

    def train(self, train_loader: DataLoader, dev_loader: DataLoader = None,
              num_epochs=20, log_steps=100, eval_steps=0, save_path="best_model.pdparams"):
        paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
        self.model.train()
        global_step = 0
        num_training_steps = num_epochs * len(train_loader)

        for epoch in range(num_epochs):
            total_loss = 0.0

            # 使用 tqdm 显示 epoch 内 batch 进度
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

            for step, (imgs, labels) in enumerate(pbar):
                preds = self.model(imgs)
                loss = self.loss_fn(preds, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                # 记录 step loss
                step_loss = loss.item()  # 改用 item() 获取标量
                self.train_step_losses.append((global_step, step_loss))
                total_loss += step_loss

                # 更新 tqdm 显示
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "global_step": global_step})

                # 验证
                if eval_steps > 0 and dev_loader and \
                        (global_step % eval_steps == 0 or global_step == num_training_steps - 1):
                    dev_score, dev_loss = self.evaluate(dev_loader, global_step)
                    print(f"[Evaluate] Step {global_step}: Dev Acc: {dev_score:.4f}, Dev Loss: {dev_loss:.4f}")

                    # 保存最优模型
                    if dev_score > self.best_score:
                        self.best_score = dev_score
                        self.save_model(save_path)
                        print(f"Saved best model with Dev Acc: {dev_score:.4f}")

                    self.model.train()  # 切回训练模式

                global_step += 1

            # epoch loss
            epoch_loss = total_loss / len(train_loader)
            self.train_epoch_losses.append(epoch_loss)
            print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")

        print(f"Training finished. Best Dev Acc: {self.best_score:.4f}")

    @paddle.no_grad()
    def evaluate(self, dev_loader: DataLoader, global_step=-1):
        self.model.eval()
        total_loss = 0.0
        if self.metric:
            self.metric.reset()

        for imgs, labels in dev_loader:
            preds = self.model(imgs)
            loss = self.loss_fn(preds, labels).item()  # 改用 item()
            total_loss += loss
            if self.metric:
                self.metric.update(preds, labels)

        dev_loss = total_loss / len(dev_loader)
        dev_score = self.metric.accumulate() if self.metric else 0.0

        # 记录
        if global_step != -1:
            self.dev_losses.append((global_step, dev_loss))
            self.dev_scores.append((global_step, dev_score))

        return dev_score, dev_loss

    @paddle.no_grad()
    def predict(self, imgs):
        self.model.eval()
        preds = self.model(imgs)
        return preds

    def save_model(self, save_path):
        paddle.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        state_dict = paddle.load(model_path)
        self.model.set_state_dict(state_dict)
