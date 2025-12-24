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
