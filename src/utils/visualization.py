import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.patches import Rectangle
from PIL import Image
import seaborn as sns


def plot_training_history(history, figsize=(15, 10)):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 学习率曲线
    if 'learning_rates' in history:
        axes[0, 2].plot(epochs, history['learning_rates'], 'g-')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)

    # 损失差值
    if len(history['train_loss']) > 1:
        loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
        axes[1, 0].plot(epochs, loss_diff, 'purple')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Train - Val Loss')
        axes[1, 0].set_title('Loss Gap (Train - Validation)')
        axes[1, 0].grid(True, alpha=0.3)

    # 准确率差值
    if len(history['train_acc']) > 1:
        acc_diff = np.array(history['val_acc']) - np.array(history['train_acc'])
        axes[1, 1].plot(epochs, acc_diff, 'orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val - Train Acc (%)')
        axes[1, 1].set_title('Accuracy Gap (Validation - Train)')
        axes[1, 1].grid(True, alpha=0.3)

    # 空子图，用于其他可视化
    axes[1, 2].axis('off')

    plt.tight_layout()
    return fig


def visualize_attention(image_path, model, transform, device, figsize=(15, 5)):
    """可视化注意力图"""
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    original_img = np.array(img)

    # 预处理
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 获取注意力图
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_attention_maps'):
            attention_maps = model.get_attention_maps(input_tensor)
        else:
            # 如果没有专门的注意力方法，使用特征图
            features = []

            def hook_fn(module, input, output):
                features.append(output)

            # 注册钩子
            hooks = []
            for name, module in model.named_modules():
                if 'se' in name.lower() or 'attention' in name.lower():
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)

            _ = model(input_tensor)

            # 移除钩子
            for hook in hooks:
                hook.remove()

            attention_maps = features

    # 可视化
    fig, axes = plt.subplots(1, min(4, len(attention_maps)) + 1, figsize=figsize)

    # 原始图像
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 注意力图
    for i, att_map in enumerate(attention_maps[:4]):
        if i >= len(axes) - 1:
            break

        att_map = att_map.squeeze().cpu().numpy()

        if len(att_map.shape) == 3:
            att_map = att_map.mean(axis=0)

        # 上采样到原始图像大小
        att_map_resized = cv2.resize(att_map, (original_img.shape[1], original_img.shape[0]))

        # 归一化
        att_map_resized = (att_map_resized - att_map_resized.min()) / (
                    att_map_resized.max() - att_map_resized.min() + 1e-8)

        # 叠加显示
        axes[i + 1].imshow(original_img, alpha=0.7)
        im = axes[i + 1].imshow(att_map_resized, cmap='jet', alpha=0.3)
        axes[i + 1].set_title(f'Attention Map {i + 1}')
        axes[i + 1].axis('off')

        # 添加颜色条
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def visualize_predictions(test_results, class_names, num_samples=12, figsize=(20, 15)):
    """可视化预测结果"""
    predictions = test_results['predictions']
    true_labels = test_results['true_labels']
    probabilities = test_results['probabilities']
    image_paths = test_results['image_paths']

    # 随机选择样本
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()

    for idx, ax_idx in enumerate(indices):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # 加载图像
        img_path = image_paths[ax_idx]
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')

        # 预测信息
        pred_class = predictions[ax_idx]
        true_class = true_labels[ax_idx]
        confidence = probabilities[ax_idx].max()

        # 颜色编码
        color = 'green' if pred_class == true_class else 'red'

        # 标题
        title = f"True: {class_names[true_class]}\n"
        title += f"Pred: {class_names[pred_class]}\n"
        title += f"Conf: {confidence:.2f}"

        ax.set_title(title, color=color, fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Model Predictions (Green=Correct, Red=Wrong)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def visualize_data_distribution(data_loader, class_names, figsize=(12, 6)):
    """可视化数据分布"""
    class_counts = np.zeros(len(class_names))

    for _, labels, _ in data_loader:
        for label in labels.numpy():
            class_counts[label] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 条形图
    bars = ax1.bar(range(len(class_names)), class_counts)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')

    # 添加数值标签
    for bar, count in zip(bars, class_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(class_counts) * 0.01,
                 f'{int(count)}', ha='center', va='bottom')

    # 饼图
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax2.pie(class_counts, labels=class_names,
                                       autopct='%1.1f%%', colors=colors,
                                       startangle=90)
    ax2.set_title('Class Proportion')

    # 调整文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.tight_layout()
    return fig


def visualize_feature_space(features, labels, class_names, figsize=(12, 10)):
    """可视化特征空间（使用t-SNE）"""
    from sklearn.manifold import TSNE

    # 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=figsize)

    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=labels, cmap='tab20', s=20, alpha=0.7)

    # 添加类别标签
    for i, class_name in enumerate(class_names):
        # 找到该类别的中心点
        class_points = features_2d[labels == i]
        if len(class_points) > 0:
            center = class_points.mean(axis=0)
            plt.annotate(class_name, center, fontsize=9, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor='white', alpha=0.8))

    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_attention_comparison(image_paths, models, model_names, transform, device, figsize=(20, 10)):
    """创建注意力机制对比图"""
    num_images = min(4, len(image_paths))
    num_models = len(models)

    fig, axes = plt.subplots(num_images, num_models + 1, figsize=figsize)

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for img_idx, img_path in enumerate(image_paths[:num_images]):
        # 加载原始图像
        img = Image.open(img_path).convert('RGB')
        original_img = np.array(img)

        # 显示原始图像
        axes[img_idx, 0].imshow(original_img)
        axes[img_idx, 0].set_title('Original')
        axes[img_idx, 0].axis('off')

        # 预处理
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 对每个模型获取注意力图
        for model_idx, (model, model_name) in enumerate(zip(models, model_names)):
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'get_attention_maps'):
                    attention_maps = model.get_attention_maps(input_tensor)
                    if len(attention_maps) > 0:
                        att_map = attention_maps[-1].squeeze().cpu().numpy()

                        if len(att_map.shape) == 3:
                            att_map = att_map.mean(axis=0)

                        # 上采样
                        att_map_resized = cv2.resize(att_map, (original_img.shape[1], original_img.shape[0]))
                        att_map_resized = (att_map_resized - att_map_resized.min()) / (
                                    att_map_resized.max() - att_map_resized.min() + 1e-8)

                        # 显示
                        axes[img_idx, model_idx + 1].imshow(original_img, alpha=0.7)
                        im = axes[img_idx, model_idx + 1].imshow(att_map_resized, cmap='jet', alpha=0.3)
                        axes[img_idx, model_idx + 1].set_title(f'{model_name}')
                        axes[img_idx, model_idx + 1].axis('off')

    plt.suptitle('Attention Mechanism Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig