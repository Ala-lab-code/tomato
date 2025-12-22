import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, roc_auc_score)
import torch
from torchmetrics import (Accuracy, Precision, Recall, F1Score,
                          ConfusionMatrix, AUROC)


class MetricsCalculator:
    """指标计算器"""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]

        # 初始化PyTorch Metrics
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='macro')
        self.precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro')

    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """计算所有指标"""
        metrics = {}

        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 每个类别的指标
        metrics['per_class'] = {}
        for i in range(self.num_classes):
            class_name = self.class_names[i]
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                class_prec = precision_score(y_true == i, y_pred == i, zero_division=0)
                class_rec = recall_score(y_true == i, y_pred == i, zero_division=0)
                class_f1 = f1_score(y_true == i, y_pred == i, zero_division=0)

                metrics['per_class'][class_name] = {
                    'accuracy': class_acc,
                    'precision': class_prec,
                    'recall': class_rec,
                    'f1': class_f1,
                    'support': mask.sum()
                }

        # ROC-AUC
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

                # 每个类别的ROC-AUC
                metrics['per_class_auc'] = {}
                for i in range(self.num_classes):
                    class_name = self.class_names[i]
                    try:
                        class_auc = roc_auc_score(y_true == i, y_prob[:, i])
                        metrics['per_class_auc'][class_name] = class_auc
                    except:
                        metrics['per_class_auc'][class_name] = 0.0
            except:
                metrics['roc_auc'] = 0.0
                metrics['per_class_auc'] = {name: 0.0 for name in self.class_names}

        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return metrics

    def compute_torch_metrics(self, predictions, targets, probs=None):
        """使用PyTorch Metrics计算指标"""
        predictions_tensor = torch.tensor(predictions)
        targets_tensor = torch.tensor(targets)

        metrics = {}

        # 计算指标
        metrics['accuracy'] = self.accuracy(predictions_tensor, targets_tensor).item()
        metrics['precision'] = self.precision(predictions_tensor, targets_tensor).item()
        metrics['recall'] = self.recall(predictions_tensor, targets_tensor).item()
        metrics['f1'] = self.f1(predictions_tensor, targets_tensor).item()

        # 混淆矩阵
        cm = self.confusion_matrix(predictions_tensor, targets_tensor)
        metrics['confusion_matrix'] = cm.numpy()

        # ROC-AUC
        if probs is not None:
            probs_tensor = torch.tensor(probs)
            try:
                metrics['roc_auc'] = self.auroc(probs_tensor, targets_tensor).item()
            except:
                metrics['roc_auc'] = 0.0

        return metrics

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', figsize=(12, 10)):
        """绘制混淆矩阵"""
        plt.figure(figsize=figsize)

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        return plt.gcf()

    def plot_roc_curves(self, y_true, y_prob, figsize=(12, 10)):
        """绘制ROC曲线"""
        plt.figure(figsize=figsize)

        # 为每个类别计算ROC曲线
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        return plt.gcf()

    def plot_metrics_comparison(self, metrics_dict, metric_name='accuracy', figsize=(10, 6)):
        """绘制指标对比图"""
        plt.figure(figsize=figsize)

        models = list(metrics_dict.keys())
        values = [metrics_dict[model][metric_name] for model in models]

        bars = plt.bar(models, values)
        plt.xlabel('Model')
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} Comparison')
        plt.ylim(0, 1.0)

        # 在柱子上添加数值
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt.gcf()

    def generate_report(self, metrics, model_name='Model'):
        """生成详细的评估报告"""
        report = f"{'=' * 60}\n"
        report += f"{model_name} Evaluation Report\n"
        report += f"{'=' * 60}\n\n"

        # 总体指标
        report += "Overall Metrics:\n"
        report += f"  Accuracy:  {metrics.get('accuracy', 0):.4f}\n"
        report += f"  Precision: {metrics.get('precision', 0):.4f}\n"
        report += f"  Recall:    {metrics.get('recall', 0):.4f}\n"
        report += f"  F1-Score:  {metrics.get('f1', 0):.4f}\n"

        if 'roc_auc' in metrics:
            report += f"  ROC-AUC:   {metrics.get('roc_auc', 0):.4f}\n"

        report += "\n"

        # 每个类别的指标
        if 'per_class' in metrics:
            report += "Per-Class Metrics:\n"
            for class_name, class_metrics in metrics['per_class'].items():
                report += f"  {class_name}:\n"
                report += f"    Accuracy:  {class_metrics['accuracy']:.4f}\n"
                report += f"    Precision: {class_metrics['precision']:.4f}\n"
                report += f"    Recall:    {class_metrics['recall']:.4f}\n"
                report += f"    F1-Score:  {class_metrics['f1']:.4f}\n"
                report += f"    Support:   {class_metrics['support']}\n"

            report += "\n"

        # 分类报告
        if 'y_true' in metrics and 'y_pred' in metrics:
            report += "Classification Report:\n"
            cr = classification_report(metrics['y_true'], metrics['y_pred'],
                                       target_names=self.class_names)
            report += cr

        return report


class FineGrainedMetrics:
    """细粒度分类专用指标"""

    @staticmethod
    def compute_similarity_confusion(confusion_matrix, similar_classes):
        """
        计算相似类别间的混淆度

        Args:
            confusion_matrix: 混淆矩阵
            similar_classes: 相似类别对的列表，如[(0,1), (2,3)]
        """
        results = {}

        for class_i, class_j in similar_classes:
            # 计算类别i被误判为类别j的比例
            total_i = confusion_matrix[class_i].sum()
            misclassified_as_j = confusion_matrix[class_i, class_j]

            if total_i > 0:
                confusion_rate = misclassified_as_j / total_i
                results[f'class_{class_i}_to_{class_j}'] = confusion_rate

        return results

    @staticmethod
    def analyze_hard_samples(predictions, probabilities, true_labels,
                             image_paths, top_k=10):
        """分析最难分类的样本"""
        # 计算每个样本的预测置信度
        correct_mask = predictions == true_labels
        confidences = probabilities.max(axis=1)

        # 最难的正样本（正确分类但置信度低）
        hard_correct_indices = np.where(correct_mask)[0]
        if len(hard_correct_indices) > 0:
            hard_correct_conf = confidences[hard_correct_indices]
            hard_correct_sorted = np.argsort(hard_correct_conf)[:top_k]
            hard_correct_samples = [(image_paths[hard_correct_indices[i]],
                                     hard_correct_conf[hard_correct_indices[i]])
                                    for i in hard_correct_sorted]
        else:
            hard_correct_samples = []

        # 最难的负样本（错误分类但置信度高）
        hard_incorrect_indices = np.where(~correct_mask)[0]
        if len(hard_incorrect_indices) > 0:
            hard_incorrect_conf = confidences[hard_incorrect_indices]
            hard_incorrect_sorted = np.argsort(hard_incorrect_conf)[-top_k:]
            hard_incorrect_samples = [(image_paths[hard_incorrect_indices[i]],
                                       hard_incorrect_conf[hard_incorrect_indices[i]])
                                      for i in hard_incorrect_sorted]
        else:
            hard_incorrect_samples = []

        return {
            'hard_correct': hard_correct_samples,
            'hard_incorrect': hard_incorrect_samples
        }