import matplotlib.pyplot as plt


def plot_hyperparam_results(results, metric="val_acc"):
    """
    绘制超参数调优结果曲线
    """
    plt.figure(figsize=(12, 6))

    # 遍历每组超参数组合
    for res in results:
        lr = res["lr"]
        dropout = res["dropout"]
        if metric == "train_loss":
            y = res["train_loss"]
        elif metric == "train_acc":
            y = res["train_acc"]
        elif metric == "val_loss":
            y = res["val_loss"]
        elif metric == "val_acc":
            y = res["val_acc"]
        else:
            raise ValueError(f"Unknown metric: {metric}")

        label = f"lr={lr}, dropout={dropout}"
        plt.plot(range(1, len(y) + 1), y, label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import paddle

# ======================================
# 1️⃣ 计算指标表格
# ======================================
def compute_metrics(models, model_names, test_loader, class_names=None, weighted=True):
    """
    返回每个模型在测试集上的指标表格

    返回 dict:
    metrics_dict = {
        "Model1": {"accuracy":..., "f1":..., "precision":..., "recall":...},
        "Model2": ...
    }
    """
    metrics_dict = {}

    # 准备测试数据
    X_list, y_list = [], []
    for imgs, labels in test_loader:
        X_list.append(imgs.numpy())
        y_list.append(labels.numpy())
    X_test = np.vstack(X_list)
    y_test = np.concatenate(y_list)

    for model, name in zip(models, model_names):
        # Paddle 模型
        if hasattr(model, 'eval'):
            model.eval()
            all_preds = []
            with paddle.no_grad():
                for imgs, _ in test_loader:
                    logits = model(imgs)
                    preds = paddle.argmax(logits, axis=1)
                    all_preds.extend(preds.numpy())
            y_pred = np.array(all_preds)
        # sklearn 模型
        else:
            y_pred = model.predict(X_test)

        avg_type = 'weighted' if weighted else 'macro'
        metrics_dict[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average=avg_type),
            "precision": precision_score(y_test, y_pred, average=avg_type),
            "recall": recall_score(y_test, y_pred, average=avg_type)
        }

    return metrics_dict

# ======================================
# 2️⃣ 画混淆矩阵
# ======================================
def plot_confusion_matrix(model, test_loader, model_name="Model", class_names=None):
    # 准备测试数据
    X_list, y_list = [], []
    for imgs, labels in test_loader:
        X_list.append(imgs.numpy())
        y_list.append(labels.numpy())
    X_test = np.vstack(X_list)
    y_test = np.concatenate(y_list)

    # 预测
    if hasattr(model, 'eval'):
        model.eval()
        all_preds = []
        with paddle.no_grad():
            for imgs, _ in test_loader:
                logits = model(imgs)
                preds = paddle.argmax(logits, axis=1)
                all_preds.extend(preds.numpy())
        y_pred = np.array(all_preds)
    else:
        y_pred = model.predict(X_test)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# ======================================
# 指标对比柱状图
# ======================================
def plot_metrics_bar(metrics_dict):
    metrics = ["accuracy", "f1", "precision", "recall"]
    model_names = list(metrics_dict.keys())
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    for i, name in enumerate(model_names):
        values = [metrics_dict[name][m] for m in metrics]
        ax.bar(x + i*width, values, width, label=name)
    ax.set_xticks(x + width/len(model_names))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend()
    plt.show()


