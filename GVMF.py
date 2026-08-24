import torch
import torch.nn as nn
import torchvision.models as models
import random
from scipy.interpolate import griddata

class CNN_Phy_dir(nn.Module):
    def __init__(self, phy_dim=12, cnn_out_channels=256):
        super().__init__()

        backbone = models.resnet18()
        # 输出的特征图维度为 (Batch, 512, 7, 7)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 将 7x7 压缩为 1x1
        self.flat = nn.Flatten()  # 展平为 512 维向量

        self.phy_pre = nn.Sequential(
            nn.Linear(phy_dim, 256),
            nn.ReLU(),

        )
        self.cnn_pre = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.fc_head = nn.Sequential(
            nn.Linear(cnn_out_channels + 256, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 2),
            nn.Sigmoid()  # 输出归一化到 [0, 1] 空间，方便回归
        )

    def forward(self, img, phy_feat):
        x_img = self.flat(self.avgpool(self.backbone(img)))
        x_phy = self.phy_pre(phy_feat)
        x_img = self.cnn_pre(x_img)
        fused = torch.cat([x_img, x_phy], dim=-1)
        return self.fc_head(fused)
