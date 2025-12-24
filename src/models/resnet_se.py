import paddle
import paddle.nn as nn
from paddle.vision.models import resnet50  # 导入 Paddle 预训练 ResNet50 模型

# -----------------------------
# 定义 SEBlock（Squeeze-and-Excitation Block）
# -----------------------------
class SEBlock(nn.Layer):
    """
    Squeeze-and-Excitation Block
    作用：对卷积特征通道进行自适应加权，强化关键特征通道
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 全局平均池化，将每个通道的 HxW 特征压缩成一个数，提取全局通道信息
        self.avg_pool = nn.AdaptiveAvgPool2D(1)  # 输出尺寸 [B, C, 1, 1]

        # 全连接层序列，实现通道权重计算
        # 1. 压缩通道 -> channels // reduction
        # 2. ReLU 激活
        # 3. 再映射回原通道数 -> channels
        # 4. Sigmoid 将权重归一化到 [0,1]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # 全局池化
        y = self.avg_pool(x).reshape([b, c])
        # 通过全连接层计算通道权重
        y = self.fc(y).reshape([b, c, 1, 1])
        # 将输入特征与通道权重相乘，实现通道重标定
        return x * y

# -----------------------------
# 定义 ResNet50 + SE 模型
# -----------------------------
class ResNet50_SE(nn.Layer):
    """
    ResNet50 + SE-Block 模型
    用于番茄叶片病害分类任务
    """
    def __init__(self, num_classes=10, pretrained=True, dropout_rate=0.5):
        super().__init__()
        # 1. 加载 PaddleVision 的预训练 ResNet50
        backbone = resnet50(pretrained=pretrained)

        # 2. 去掉原始 ResNet50 的最后全连接层 fc
        # 只保留卷积层 + 最后的 avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出 [B,2048,1,1]

        # 3. 添加 SE 注意力模块
        # 对 backbone 输出的 2048 通道进行通道重标定
        self.se = SEBlock(2048)

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # 防止过拟合
            nn.Linear(512, num_classes) # 防止过拟合
        )

    def forward(self, x):
        # 输入 x: [B, 3, 224, 224]

        # 1. backbone 提取卷积特征
        # 输出尺寸 [B, 2048, 1, 1] 因为最后有 avgpool
        x = self.backbone(x)

        # 2. SE 注意力模块，强化关键通道
        x = self.se(x)

        # 3. 分类头
        x = self.classifier(x)  # 输出 [B, num_classes]

        return x

