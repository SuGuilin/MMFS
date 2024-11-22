import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y  # Channel-wise scaling


class FeaLearning(nn.Module):
    def __init__(self, num_channels=64, reduction=16 ):
        super().__init__()
        
        # Convolution layers for denoising
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        # Feature enhancement
        self.se_rgb = SE(num_channels)
        self.se_thermal = SE(num_channels)

        # Global channel regulation layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(4 * num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 2 * num_channels)
        )

        # Local spatial rectification layers
        self.initial_conv = nn.Conv2d(num_channels * 2, num_channels, kernel_size=1)
        self.conv1_rect = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2_rect = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=2, padding=2)
        self.final_conv = nn.Conv2d(num_channels, num_channels * 2, kernel_size=1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def cosine_denoising(self, rgb_features, thermal_features):
        rgb_out = self.conv4(self.conv2(self.conv1(rgb_features)) + self.conv3(self.conv1(rgb_features)))
        thermal_out = self.conv4(self.conv2(self.conv1(thermal_features)) + self.conv3(self.conv1(thermal_features)))

        rgb_out = self.se_rgb(rgb_out)
        thermal_out = self.se_thermal(thermal_out)

        # Normalize feature maps
        normalized_rgb = F.normalize(rgb_out.reshape(rgb_out.size(0), -1), p=2, dim=1)
        normalized_thermal = F.normalize(thermal_out.reshape(thermal_out.size(0), -1), p=2, dim=1)

        # Calculate mean and center features
        mean_rgb = normalized_rgb.mean(dim=1, keepdim=True)
        mean_thermal = normalized_thermal.mean(dim=1, keepdim=True)
        centered_rgb = normalized_rgb - mean_rgb
        centered_thermal = normalized_thermal - mean_thermal

        # Calculate cosine similarity
        similarity = 1 - (centered_rgb * centered_thermal).sum(dim=1) / (
            centered_rgb.norm(dim=1) * centered_thermal.norm(dim=1) + 1e-6
        )
        similarity = similarity.view(rgb_out.size(0), 1, 1, 1)
        thermal_features += thermal_features * similarity
        return rgb_out, thermal_features

    def global_channel_regulation(self, rgb_features, thermal_features):
        batch_size, channels, _, _ = rgb_features.shape
        combined_features = torch.cat((rgb_features, thermal_features), dim=1)

        # Obtain global features
        avg_pool = self.global_avg_pool(combined_features)
        max_pool = self.global_max_pool(combined_features)

        # Flatten and pass through MLP
        channel_weights = self.mlp(torch.cat((avg_pool, max_pool), dim=1).view(combined_features.size(0), -1))
        rgb_weights, thermal_weights = channel_weights.chunk(2, dim=1)

        # Apply weights
        regulated_rgb = rgb_features * rgb_weights.view(batch_size, channels, 1, 1)
        regulated_thermal = thermal_features * thermal_weights.view(batch_size, channels, 1, 1)

        return regulated_rgb, regulated_thermal

    def local_spatial_rectification(self, regulated_rgb, regulated_thermal):
        combined_features = torch.cat((regulated_rgb, regulated_thermal), dim=1)
        x = self.initial_conv(combined_features)
        x = self.conv1_rect(x) + self.conv2_rect(x)
        x = self.final_conv(x)
        rgb_weights, thermal_weights = x.chunk(2, dim=1)

        return rgb_weights, thermal_weights

    def forward(self, rgb_features, thermal_features):
        rgb_out, thermal_out = self.cosine_denoising(rgb_features, thermal_features)
        regulated_rgb, regulated_thermal = self.global_channel_regulation(rgb_out, thermal_out)
        rgb_weights, thermal_weights = self.local_spatial_rectification(regulated_rgb, regulated_thermal)
        out_rgb = rgb_features + self.dropout(thermal_weights * regulated_thermal) + regulated_thermal
        out_thermal = thermal_features + self.dropout(rgb_weights * regulated_rgb) + regulated_rgb

        return out_rgb, out_thermal
    
if __name__ == "__main__":
    import cv2
    batch_size = 1
    height = 480
    width = 640
    channels = 3

    rgb = cv2.imread('/home/suguilin/MMFS/datasets/MFNet/RGB/00093D.png') / 255.
    ther = cv2.imread('/home/suguilin/MMFS/datasets/MFNet/Modal/00093D.png') / 255.

    rgb = torch.from_numpy(rgb).float()
    ther = torch.from_numpy(ther).float()

    F_rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2)
    F_ther = ther.unsqueeze(0).permute(0, 3, 1, 2)

    # F_rgb = torch.randn(batch_size, channels, height, width)  # RGB features
    # F_ther = torch.randn(batch_size, channels, height, width)  # Thermal features

    # Initialize and run the CFRM
    cfrm = FeaLearning(num_channels=channels)
    X_out_rgb, X_out_ther = cfrm(F_rgb, F_ther)

    cv2.imwrite('rgb.png', X_out_rgb.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255) 
    cv2.imwrite('ther.png', X_out_ther.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255) 

    print("Output RGB shape:", X_out_rgb.shape)
    print("Output Thermal shape:", X_out_ther.shape)