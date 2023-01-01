import torch
import torch.nn as nn
import torch.nn.functional as F


# Unet的一层、两次卷积过程
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 数据的归一化处理 使数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# Unet的一次2*2降采样 + 两次卷积过程
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Unet的一次复制剪切 + 反卷积 + 两次卷积过程
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 选择双线性插值或反卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # 反卷积后上采样
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 先反卷积
        x1 = self.up(x1)
        # 再复制、剪切（concat过程）
        # 由于左右图像大小不一样 选择对小的padding
        # 原论文是对大的crop 导致最终图像比原图像小
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        # 最后两次卷积
        return self.conv(x)


# 输出部分做两次卷积
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
