import torch
import torch.nn.functional as F

def compute_saliency(x):
    """
    计算每个像素的显著性值 Sx(k)
    x: 输入图像 (B, C, H, W)
    返回: 每个像素的显著性值 (B, H, W)
    """
    B, C, H, W = x.shape
    saliency = torch.zeros(B, H, W, device=x.device)

    for i in range(256):  # 假设像素值在[0, 255]范围内
        Hx_i = torch.histc(x, bins=256, min=0, max=255)  # 计算直方图 Hx(i)
        diff = torch.abs(x - i)  # 计算 |x(k) - i|
        saliency += Hx_i[i] * diff

    return saliency
