import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class SELayer(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNetSE(nn.Module):
    """ResNet50 with SE Attention"""

    def __init__(self, num_classes=10, pretrained=True, reduction=16):
        super(ResNetSE, self).__init__()

        # 加载预训练ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            base_model = resnet50(weights=weights)
        else:
            base_model = resnet50(weights=None)

        # 获取特征提取层
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # 添加SE注意力模块
        self.se1 = SELayer(256, reduction)
        self.se2 = SELayer(512, reduction)
        self.se3 = SELayer(1024, reduction)
        self.se4 = SELayer(2048, reduction)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # 梯度裁剪
        self.grad_norm = 1.0

    def forward(self, x, return_attention=False):
        # 提取特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差块 + SE注意力
        x = self.layer1(x)
        x = self.se1(x)

        x = self.layer2(x)
        x = self.se2(x)

        x = self.layer3(x)
        x = self.se3(x)

        x = self.layer4(x)
        x = self.se4(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 分类
        output = self.fc(x)

        if return_attention:
            # 返回最后SE层的注意力权重
            with torch.no_grad():
                b, c, _, _ = self.se4.input_tensor.shape if hasattr(self.se4, 'input_tensor') else (1, 2048, 7, 7)
                attention = self.se4.fc(self.se4.avgpool(self.se4.input_tensor).view(b, c))
                attention = attention.view(b, c, 1, 1)
            return output, attention
        else:
            return output

    def get_attention_maps(self, x):
        """获取注意力图"""
        self.eval()
        with torch.no_grad():
            # 前向传播并保存中间特征
            attentions = []

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            se1_out = self.se1(x)
            attentions.append(se1_out.mean(dim=1, keepdim=True))
            x = se1_out

            x = self.layer2(x)
            se2_out = self.se2(x)
            attentions.append(se2_out.mean(dim=1, keepdim=True))
            x = se2_out

            x = self.layer3(x)
            se3_out = self.se3(x)
            attentions.append(se3_out.mean(dim=1, keepdim=True))
            x = se3_out

            x = self.layer4(x)
            se4_out = self.se4(x)
            attentions.append(se4_out.mean(dim=1, keepdim=True))

            return attentions


class ResNetSEWithCBAM(nn.Module):
    """ResNet with both SE and CBAM attention"""

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        from .efficientnet_cbam import CBAM

        # 加载预训练ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
            base_model = resnet50(weights=weights)
        else:
            base_model = resnet50(weights=None)

        # 特征提取层
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # 混合注意力
        self.se = SELayer(2048)
        self.cbam = CBAM(2048, reduction_ratio=16)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.se(x)  # SE注意力
        x = self.cbam(x)  # CBAM注意力
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x