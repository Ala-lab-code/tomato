import matplotlib.pyplot as plt

def plot_hyperparam_curves(
    results,
    metric="acc",        # "acc" or "loss"
    split="val",         # "train" or "val"
    title=None,
    figsize=(10, 6)
):
    """
    绘制多组超参数组合的训练/验证曲线

    Args:
        results (list[dict]): 超参数搜索结果列表
        metric (str): "acc" 或 "loss"
        split (str): "train" 或 "val"
        title (str): 图标题
        figsize (tuple): 图像大小
    """

    assert metric in ["acc", "loss"]
    assert split in ["train", "val"]

    key = f"{split}_{metric}"

    plt.figure(figsize=figsize)

    for r in results:
        label = (
            f"lr={r['lr']}, "
            f"bs={r['batch_size']}, "
            f"freeze_l3={r['freeze_layer3']}"
        )
        plt.plot(r[key], label=label)

    if title is None:
        title = f"{split.capitalize()} {metric.upper()} Comparison"

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




