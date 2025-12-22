import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm
import numpy as np
import time
import os
from datetime import datetime


class Trainer:
    """深度学习训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, config, class_names=None):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 训练设备
            config: 训练配置
            class_names: 类别名称列表
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.class_names = class_names

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )

        # 学习率调度器
        scheduler_type = config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs']
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=5,
                factor=0.5
            )
        else:
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # 创建输出目录
        self.log_dir = config['logging']['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

        print(f"Trainer initialized on {device}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, targets, _) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 梯度清零
            self.optimizer.zero_grad()

            # 混合精度训练
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        # 计算epoch指标
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets, _ in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 收集预测结果
                probs = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc, all_predictions, all_targets, all_probs

    def train(self, epochs):
        """训练模型"""
        print(f"\n{'=' * 60}")
        print(f"Starting training for {epochs} epochs")
        print(f"{'=' * 60}")

        best_val_acc = 0.0
        patience_counter = 0
        patience = self.config['training'].get('patience', 15)

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc, _, _, _ = self.validate()

            # 更新学习率
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # 打印结果
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.1f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(f'best_model.pth', epoch, val_acc)
                patience_counter = 0
                print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"↻ No improvement ({patience_counter}/{patience})")

            # 定期保存检查点
            if epoch % self.config['logging'].get('save_freq', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_acc)

            # 早停
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print(f"\n{'=' * 60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'=' * 60}")

        return self.history

    def test(self):
        """测试模型"""
        print(f"\n{'=' * 60}")
        print(f"Testing model on test set")
        print(f"{'=' * 60}")

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []
        all_probs = []
        all_paths = []

        with torch.no_grad():
            for inputs, targets, paths in tqdm(self.test_loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 收集结果
                probs = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(paths)

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Correct/Total: {correct}/{total}")

        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_targets),
            'probabilities': np.array(all_probs),
            'image_paths': all_paths
        }

    def save_checkpoint(self, filename, epoch, val_acc):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config,
            'class_names': self.class_names
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        filepath = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.history = checkpoint['history']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")

        return checkpoint['epoch']


class BaselineTrainer:
    """基线模型训练器"""

    def __init__(self, config):
        self.config = config

    def run(self, data_dir):
        """运行基线模型训练"""
        from src.data.dataset import create_dataloaders
        from src.models.baseline import LogisticRegressionBaseline

        print("Running baseline model...")

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            data_dir,
            batch_size=32,
            num_workers=2
        )

        # 创建基线模型
        input_size = 224 * 224 * 3
        baseline = LogisticRegressionBaseline(
            input_size=input_size,
            num_classes=len(class_names),
            C=self.config['training']['C'],
            solver=self.config['training']['solver'],
            max_iter=self.config['training']['max_iter']
        )

        # 训练模型
        train_acc = baseline.train(train_loader)

        # 评估模型
        results = baseline.evaluate(test_loader, class_names)

        # 保存模型
        if self.config['logging']['save_model']:
            model_path = os.path.join(self.config['logging']['log_dir'], 'baseline_model.pkl')
            baseline.save_model(model_path)

        return results, class_names