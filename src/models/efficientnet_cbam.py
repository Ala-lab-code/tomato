import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class EfficientNetCBAM(nn.Module):
    """EfficientNet with CBAM Attention"""

    def __init__(self, num_classes=10, pretrained=True, model_name='efficientnet_b0'):
        super(EfficientNetCBAM, self).__init__()

        # 加载预训练EfficientNet
        if pretrained:
            if model_name == 'efficientnet_b0':
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                base_model = efficientnet_b0(weights=weights)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        else:
            base_model = efficientnet_b0(weights=None)

        # 获取特征提取层
        self.features = base_model.features

        # 在关键层添加CBAM注意力
        self.cbam1 = CBAM(32, reduction_ratio=8)  # 早期特征
        self.cbam2 = CBAM(112, reduction_ratio=16)  # 中期特征
        self.cbam3 = CBAM(320, reduction_ratio=32)  # 深层特征

        # 自适应池化和分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(320, num_classes)
        )

        # 存储中间特征用于可视化
        self.feature_maps = {}

    def forward(self, x, return_features=False):
        # 提取特征
        features = []

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 2:  # 第一层后
                x = self.cbam1(x)
                features.append(x)
            elif idx == 4:  # 第三层后
                x = self.cbam2(x)
                features.append(x)
            elif idx == len(self.features) - 1:  # 最后一层
                x = self.cbam3(x)
                features.append(x)

        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 分类
        output = self.classifier(x)

        if return_features:
            return output, features
        else:
            return output

    def get_attention_maps(self, x):
        """获取注意力图"""
        self.eval()
        with torch.no_grad():
            attention_maps = []

            # 前向传播
            for idx, layer in enumerate(self.features):
                x = layer(x)
                if idx == 2:  # CBAM1后
                    spatial_att = self.cbam1.spatial_attention(x)
                    attention_maps.append(spatial_att)
                elif idx == 4:  # CBAM2后
                    spatial_att = self.cbam2.spatial_attention(x)
                    attention_maps.append(spatial_att)
                elif idx == len(self.features) - 1:  # CBAM3后
                    spatial_att = self.cbam3.spatial_attention(x)
                    attention_maps.append(spatial_att)

            return attention_maps


class EfficientNetSE(nn.Module):
    """EfficientNet with SE Attention"""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        from .resnet_se import SELayer

        # 加载预训练EfficientNet
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = efficientnet_b0(weights=weights)
        else:
            base_model = efficientnet_b0(weights=None)

        # 特征提取
        self.features = base_model.features

        # 在关键层添加SE注意力
        self.se1 = SELayer(32, reduction=8)
        self.se2 = SELayer(112, reduction=16)
        self.se3 = SELayer(320, reduction=32)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(320, num_classes)
        )

    def forward(self, x):
        # 提取特征并应用SE注意力
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 2:  # 第一层后
                x = self.se1(x)
            elif idx == 4:  # 第三层后
                x = self.se2(x)
            elif idx == len(self.features) - 1:  # 最后一层
                x = self.se3(x)

        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x