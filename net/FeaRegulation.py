import torch
import torch.nn as nn
import torch.nn.functional as F

class COS_Denoise(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        # self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        # self.conv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        # self.conv1_4 = nn.Conv2d(in_channels, 1, kernel_size=1)

        # self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        # self.conv2_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        # self.conv2_4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, F_rgb, F_ther):
        F_rgb = self.conv4(self.conv2(self.conv1(F_rgb)) + self.conv3(self.conv1(F_rgb)))
        F_ther = self.conv4(self.conv2(self.conv1(F_ther)) + self.conv3(self.conv1(F_ther)))

        # Normalize feature maps
        A_rgb = F.normalize(F_rgb.view(F_rgb.size(0), -1), p=2, dim=1)
        A_ther = F.normalize(F_ther.view(F_ther.size(0), -1), p=2, dim=1)

        # Calculate mean
        mean_rgb = A_rgb.mean(dim=1, keepdim=True)
        mean_ther = A_ther.mean(dim=1, keepdim=True)

        # Center the features
        A_rgb_centered = A_rgb - mean_rgb
        A_ther_centered = A_ther - mean_ther

        # Calculate cosine similarity
        similarity = 1 - (A_rgb_centered * A_ther_centered).sum(dim=1) / (
            A_rgb_centered.norm(dim=1) * A_ther_centered.norm(dim=1) + 1e-6
        )

        similarity = similarity.view(F_rgb.size(0), 1, 1, 1)
        F_ther += F_ther * similarity
        return F_rgb, F_ther


class GlobalChannelRegulation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(4 * in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 2 * in_channels)
        )

    def forward(self, F_rgb, F_ther):
        # Concatenate features
        b, c, _, _ = F_rgb.shape
        combined = torch.cat((F_rgb, F_ther), dim=1)

        # Obtain global features
        avg_pool = self.global_avg_pool(combined)
        max_pool = self.global_max_pool(combined)

        # Flatten and pass through MLP
        channel_weights = self.mlp(torch.cat((avg_pool, max_pool), dim=1).view(combined.size(0), -1))
        
        # Split channel weights back
        W_rgb, W_ther = channel_weights.chunk(2, dim=1)

        # Apply weights
        X_C_rgb = F_rgb * W_rgb.view(b, c, 1, 1)
        X_C_ther = F_ther * W_ther.view(b, c, 1, 1)

        return X_C_rgb, X_C_ther


class LocalSpatialRectification(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)

    def forward(self, X_C_rgb, X_C_ther):
        # Concatenate features
        combined = torch.cat((X_C_rgb, X_C_ther), dim=1)

        # Apply convolutions
        x = self.conv1(combined)
        x = self.conv2_1(x) + self.conv2_2(x)
        x = self.conv3(x)
        W_rgb, W_ther = x.chunk(2, dim=1)

        return W_rgb, W_ther


class CrossFeatureRegulationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cos_denoise = COS_Denoise()
        self.global_channel_regulation = GlobalChannelRegulation(in_channels)
        self.local_spatial_rectification = LocalSpatialRectification(in_channels)

    def forward(self, F_rgb, F_ther):
        # Compute similarity scores
        F_rgb, F_ther = self.cos_denoise(F_rgb, F_ther)

        # Global channel regulation
        X_C_rgb, X_C_ther = self.global_channel_regulation(F_rgb, F_ther)

        X_C_rgb, X_C_ther = F_rgb * X_C_rgb, F_ther * X_C_ther
        # Local spatial rectification
        W_rgb, W_ther = self.local_spatial_rectification(X_C_rgb, X_C_ther)

        # Final output
        X_out_rgb = F_rgb + W_ther * X_C_ther + X_C_ther
        X_out_ther = F_ther + W_rgb * X_C_rgb + X_C_rgb

        return X_out_rgb, X_out_ther


if __name__ == "__main__":
    import cv2
    batch_size = 4
    height = 32
    width = 32
    channels = 64

    rgb = cv2.imread('/home/suguilin/MMFS/datasets/MFNet/RGB/00093D.png', 0) / 255.
    ther = cv2.imread('/home/suguilin/MMFS/datasets/MFNet/Modal/00093D.png', 0) / 255.

    F_rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2)
    F_ther = ther.unsqueeze(0).permute(0, 3, 1, 2)

    # F_rgb = torch.randn(batch_size, channels, height, width)  # RGB features
    # F_ther = torch.randn(batch_size, channels, height, width)  # Thermal features

    # Initialize and run the CFRM
    cfrm = CrossFeatureRegulationModule(in_channels=channels)
    X_out_rgb, X_out_ther = cfrm(F_rgb, F_ther)

    print("Output RGB shape:", X_out_rgb.shape)
    print("Output Thermal shape:", X_out_ther.shape)