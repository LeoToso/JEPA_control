"""ResNet-style visual encoder phi_theta: (3,64,64)->z in R^d (L2-normalised)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=2):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(out_ch)
        if stride!=1 or in_ch!=out_ch:
            self.shortcut=nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=stride,bias=False),nn.BatchNorm2d(out_ch))
        else:
            self.shortcut=nn.Identity()
    def forward(self,x):
        out=F.relu(self.bn(self.conv(x)))
        return out

class VisualEncoder(nn.Module):
    """ResNet-based visual encoder.

    Architecture: 4 conv blocks Conv2d(stride=2)->BN->ReLU, channels [3->32->64->128->256],
    Global average pooling, Linear projection to latent_dim (d), L2 normalisation.
    """
    def __init__(self,latent_dim=32,channels=None,image_size=64,normalize=True):
        super().__init__()
        if channels is None:
            channels=[32,64,128,256]
        self.latent_dim=latent_dim
        self.normalize=normalize
        in_ch=3
        blocks=[]
        for out_ch in channels:
            blocks.append(ResidualBlock(in_ch,out_ch,stride=2))
            in_ch=out_ch
        self.conv_blocks=nn.Sequential(*blocks)
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Linear(channels[-1],latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        h=self.conv_blocks(x)
        h=self.gap(h).squeeze(-1).squeeze(-1)
        z=self.proj(h)
        if self.normalize:
            z=F.normalize(z,dim=-1)
        return z

    @torch.no_grad()
    def encode(self,x):
        return self(x)
