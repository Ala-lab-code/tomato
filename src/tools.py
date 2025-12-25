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



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(model, data_loader, class_names, model_type='lr'):
    """
    混淆矩阵绘制函数
    model_type: 'lr' 或 'deep'
    """
    y_true = []
    y_pred = []

    if model_type == 'lr':
        # 对于逻辑回归，data_loader 返回 (features, labels)
        for batch in data_loader:
            X_batch, y_batch = batch
            y_true.extend(y_batch)
            y_pred.extend(model.model.predict(model.scaler.transform(X_batch)) if model.pca is None
                          else model.model.predict(model.pca.transform(model.scaler.transform(X_batch))))
    elif model_type == 'deep':
        import paddle
        model.model.eval()
        for imgs, labels in data_loader:
            logits = model.model(imgs)
            preds = paddle.argmax(logits, axis=1).numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    else:
        raise ValueError("model_type must be 'lr' or 'deep'")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_type.upper()})')
    plt.show()
