  
import torch
import torch.nn as nn


class DeeplabV3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=256,
                 out_stride=16):
        super(DeeplabV3, self).__init__()

        if out_stride == 16:
            dilations = [6, 12, 18]
        elif out_stride == 8:
            dilations = [12, 24, 36]

        self.aspp = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[0], padding=dilations[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[1], padding=dilations[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilations[2], padding=dilations[2])
        ])
        self.aspp_bn = nn.BatchNorm2d(hidden_channels * 4)
        self.drop = nn.Dropout(0.3)
        self.aspp_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)


        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn =  nn.BatchNorm2d(hidden_channels)
        self.pool_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        
        self.fin_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.aspp], dim=1)
        out = self.aspp_bn(out)
        out = self.drop(out)
        out = self.aspp_conv(out)


        # Global pooling
        pool = self.global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_conv(pool)

        pool = pool.repeat(1, 1, x.size(2), x.size(3)) #to sum

        out += pool
        out = self.fin_bn(out)
        return out