import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedDenoisingModule(nn.Module):
    def __init__(self, num_channels=64):
        super().__init__()
        
        # Convolution layers for denoising
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(num_channels, num_channels, kernel_size=1)

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

    def cosine_denoising(self, rgb_features, thermal_features):
        rgb_out = self.conv4(self.conv2(self.conv1(rgb_features)) + self.conv3(self.conv1(rgb_features)))
        thermal_out = self.conv4(self.conv2(self.conv1(thermal_features)) + self.conv3(self.conv1(thermal_features)))

        # Normalize feature maps
        normalized_rgb = F.normalize(rgb_out.view(rgb_out.size(0), -1), p=2, dim=1)
        normalized_thermal = F.normalize(thermal_out.view(thermal_out.size(0), -1), p=2, dim=1)

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
        out_rgb = rgb_features + thermal_weights * regulated_thermal + regulated_thermal
        out_thermal = thermal_features + rgb_weights * regulated_rgb + regulated_rgb

        return out_rgb, out_thermal
    
