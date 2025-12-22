import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class LogisticRegressionBaseline:
    """逻辑回归基线模型"""

    def __init__(self, input_size=224 * 224 * 3, num_classes=10, C=1.0,
                 solver='lbfgs', max_iter=1000):
        """
        初始化逻辑回归模型

        Args:
            input_size: 输入特征维度
            num_classes: 类别数
            C: 正则化强度
            solver: 优化算法
            max_iter: 最大迭代次数
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            multi_class='multinomial',
            random_state=42,
            verbose=1,
            n_jobs=-1
        )
        self.is_trained = False

    def preprocess_images(self, data_loader):
        """
        预处理图像数据为逻辑回归可用的格式

        Args:
            data_loader: PyTorch数据加载器

        Returns:
            X: 特征矩阵 (n_samples, input_size)
            y: 标签向量 (n_samples,)
        """
        X_list = []
        y_list = []

        print("Preprocessing images for logistic regression...")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            # 展平图像
            images_flat = images.view(images.size(0), -1).numpy()
            X_list.append(images_flat)
            y_list.append(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches...")

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        print(f"Preprocessing complete. X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def train(self, train_loader):
        """训练逻辑回归模型"""
        print("Training logistic regression model...")

        # 预处理训练数据
        X_train, y_train = self.preprocess_images(train_loader)

        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # 计算训练准确率
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        print(f"Training completed!")
        print(f"Training accuracy: {train_acc:.4f}")

        return train_acc

    def evaluate(self, test_loader, class_names=None):
        """评估模型性能"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")

        # 预处理测试数据
        X_test, y_test = self.preprocess_images(test_loader)

        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # 计算指标
        test_acc = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 50}")
        print(f"Logistic Regression Baseline Results")
        print(f"{'=' * 50}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Number of samples: {len(y_test)}")

        # 分类报告
        if class_names:
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=class_names))
        else:
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)

        return {
            'accuracy': test_acc,
            'predictions': y_pred,
            'true_labels': y_test,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }

    def save_model(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class LogisticRegressionTorch(nn.Module):
    """PyTorch版本的逻辑回归（用于对比）"""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        return self.linear(x)