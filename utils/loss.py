import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models import vgg16
import torchvision

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = MySobelxy()
        # self.perceptualLoss = PerceptualLoss()
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)

    def forward(self, image_vis, image_ir, generate_img, config):
        # image_y = image_vis[:, :1, :, :]
        image_y = image_vis
        x_in_max = torch.max(image_y, image_ir)
        x_in_mean = (image_vis + image_ir) / 2.0
        loss_in = F.l1_loss(generate_img, x_in_max)
        bg_loss = F.l1_loss(generate_img, x_in_mean)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        YCbCr_Fuse = RGB2YCrCb(generate_img)
        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
        Cb_Fuse = YCbCr_Fuse[:,2:,:,:] 
        YCbCr_R_vis = RGB2YCrCb(image_vis) 
        Cr_R_vis = YCbCr_R_vis[:,1:2,:,:]
        Cb_R_vis = YCbCr_R_vis[:,2:,:,:] 
        color_loss = F.l1_loss(Cb_Fuse, Cb_R_vis) + F.l1_loss(Cr_Fuse, Cr_R_vis)

        ssim_loss = self.ssim_loss(image_vis, image_ir[:, 0:1, :, :], generate_img)
        
        # loss_perceptual = self.perceptualLoss(generate_img, x_in_max)
        # loss_tv = total_variation_loss(generate_img)

        loss_total = config.alpha * loss_in + config.beta * loss_grad + config.gamma * color_loss + ssim_loss#+ 3 * loss_tv+ loss_perceptual 
        return loss_total#, loss_in, loss_grad#, loss_perceptual # , loss_tv

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return sobelx, sobely

class MakeFusionLoss(nn.Module):
    def __init__(self):
        super(MakeFusionLoss, self).__init__()
        self.l1_loss =  nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11, size_average=True)

    def forward(self, input_vis, input_ir, Fuse):
        YCbCr_Fuse = RGB2YCrCb(Fuse) 
        Y_Fuse  = YCbCr_Fuse[:,0:1,:,:]
        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
        Cb_Fuse = YCbCr_Fuse[:,2:,:,:]  
        
        # R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.5, 1)
        R_vis = input_vis
        YCbCr_R_vis = RGB2YCrCb(R_vis) 
        Y_R_vis = YCbCr_R_vis[:,0:1,:,:]
        Cr_R_vis = YCbCr_R_vis[:,1:2,:,:]
        Cb_R_vis = YCbCr_R_vis[:,2:,:,:]

        Y_R_ir = input_ir[:, 0:1, :, :]
        # R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1.7)
        R_ir = input_ir

        Fuse_R = torch.unsqueeze(Fuse[:,0,:,:],1)
        Fuse_G = torch.unsqueeze(Fuse[:,1,:,:],1)
        Fuse_B = torch.unsqueeze(Fuse[:,2,:,:],1)
        Fuse_R_grad_x, Fuse_R_grad_y =   Sobelxy(Fuse_R)
        Fuse_G_grad_x, Fuse_G_grad_y =   Sobelxy(Fuse_G)
        Fuse_B_grad_x, Fuse_B_grad_y =   Sobelxy(Fuse_B)
        Fuse_grad_x = torch.cat([Fuse_R_grad_x, Fuse_G_grad_x, Fuse_B_grad_x], 1)
        Fuse_grad_y = torch.cat([Fuse_R_grad_y, Fuse_G_grad_y, Fuse_B_grad_y], 1)

        R_VIS_R = torch.unsqueeze(R_vis[:,0,:,:], 1)
        R_VIS_G = torch.unsqueeze(R_vis[:,1,:,:], 1)
        R_VIS_B = torch.unsqueeze(R_vis[:,2,:,:], 1)
        R_VIS_R_grad_x, R_VIS_R_grad_y =   Sobelxy(R_VIS_R)
        R_VIS_G_grad_x, R_VIS_G_grad_y =   Sobelxy(R_VIS_G)
        R_VIS_B_grad_x, R_VIS_B_grad_y =   Sobelxy(R_VIS_B)
        R_VIS_grad_x = torch.cat([R_VIS_R_grad_x, R_VIS_G_grad_x, R_VIS_B_grad_x], 1)
        R_VIS_grad_y = torch.cat([R_VIS_R_grad_y, R_VIS_G_grad_y, R_VIS_B_grad_y], 1)

        R_IR_R = torch.unsqueeze(R_ir[:,0,:,:],1)
        R_IR_G = torch.unsqueeze(R_ir[:,1,:,:],1)
        R_IR_B = torch.unsqueeze(R_ir[:,2,:,:],1)
        R_IR_R_grad_x,R_IR_R_grad_y =   Sobelxy(R_IR_R)
        R_IR_G_grad_x,R_IR_G_grad_y =   Sobelxy(R_IR_G)
        R_IR_B_grad_x,R_IR_B_grad_y =   Sobelxy(R_IR_B)
        R_IR_grad_x = torch.cat([R_IR_R_grad_x, R_IR_G_grad_x,R_IR_B_grad_x], 1)
        R_IR_grad_y = torch.cat([R_IR_R_grad_y, R_IR_G_grad_y,R_IR_B_grad_y], 1)

        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
        joint_int  = torch.maximum(R_vis, R_ir)
        
        con_loss = self.l1_loss(Fuse, joint_int)
        gradient_loss = 0.5 * self.l1_loss(Fuse_grad_x, joint_grad_x) + 0.5 * self.l1_loss(Fuse_grad_y, joint_grad_y)
        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)
        # print(Y_R_vis.shape, Y_R_ir.shape, Y_Fuse.shape)
        ssim_loss = self.ssim_loss(R_vis, Y_R_ir, Fuse)

        fusion_loss_total = 0.5 * con_loss  + 0.2 * gradient_loss  + 1 * color_loss + 0.1 * ssim_loss

        return fusion_loss_total

class DegradationLoss(nn.Module):
    def __init__(self):
        super(DegradationLoss, self).__init__()
        
    def smoothness_loss(self, kernel):
        dx = torch.abs(kernel[:, :, 1:, :] - kernel[:, :, :-1, :])  # x方向梯度
        dy = torch.abs(kernel[:, :, :, 1:] - kernel[:, :, :, :-1])  # y方向梯度
        return torch.mean(dx) + torch.mean(dy)
    
    def forward(self, params):
        self.b1 = params[0]
        self.b2 = params[1]
        self.c1 = params[2]
        self.c2 = params[3]
        self.n1 = params[4]
        self.n2 = params[5]
        self.j1 = params[6]
        self.j2 = params[7]
        self.blur_kernel = params[8]
        blur_kernel_loss = torch.norm(self.blur_kernel, p=1)
        blur_kernel_smoothloss = self.smoothness_loss(self.blur_kernel)
        blur_reg = blur_kernel_loss + blur_kernel_smoothloss
        brightness_reg = (self.b1 - 1.0)**2 + (self.b2 - 1.0)**2
        contrast_reg = (self.c1 - 1.0)**2 + (self.c2 - 1.0)**2
        noise_reg = self.n1**2 + self.n2**2
        jpeg_reg = (self.j1 - 75.0)**2 + (self.j2 - 75.0)**2
        regularization_loss = brightness_reg + contrast_reg + noise_reg + blur_reg + jpeg_reg
        return regularization_loss



class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window_rgb = self.create_window(window_size, self.channel)
        self.window_ir = self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.tensor([(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        device = img1.device
        window = window.to(device)
        
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img_rgb, img_ir, img_fused):
        (_, channel, _, _) = img_rgb.size()
        window_rgb = self.window_rgb.to(img_rgb.device)
        window_ir = self.window_ir.to(img_ir.device).type(img_ir.dtype)
        if channel == self.channel and window_rgb.data.type() == img_rgb.data.type():
            window_rgb = self.window_rgb
        else:
            window_rgb = self.create_window(self.window_size, channel).to(img_rgb.device).type(img_rgb.dtype)
            self.window_rgb = window_rgb
            self.channel = channel
        ssim_rgb_fused = self.ssim(img_rgb, img_fused, window_rgb, self.window_size, channel, self.size_average)
        # 将三通道的融合图像转换为灰度图，再与红外图像进行相似性指数度量
        single_img_fused = 0.2989 * img_fused[:, 0:1, :, :] + 0.5870 * img_fused[:, 1:2, :, :] + 0.1140 * img_fused[:, 2:3, :, :]
        ssim_ir_fused = self.ssim(img_ir, single_img_fused, window_ir, self.window_size, 1, self.size_average)
        return 1 - (ssim_rgb_fused + ssim_ir_fused) / 2

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features[:16]
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        self.vgg.to(img1.device)
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return torch.mean(torch.abs(feat1 - feat2))

def total_variation_loss(img, weight=1.0):
    """
    Calculate the total variation loss for an image.

    Parameters:
    img (torch.Tensor): Input image tensor of shape (B, C, H, W)
    weight (float): Weight parameter to scale the total variation loss

    Returns:
    torch.Tensor: Total variation loss
    """
    diff_i = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    diff_j = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    loss = weight * (torch.sum(diff_i) + torch.sum(diff_j))
    
    # Normalize by image size
    H, W = img.size(2), img.size(3)
    loss = loss / (H * W)
    
    return loss

def contrastive_loss(fused_image, image1, image2, temperature=0.5):
    b, c, h, w = fused_image.shape
    fused_flat = fused_image.view(b, -1)
    image1_flat = image1.view(b, -1)
    image2_flat = image2.view(b, -1)

    pos_sim = F.cosine_similarity(fused_flat, image1_flat)
    neg_sim = F.cosine_similarity(fused_flat, image2_flat)

    loss = -torch.log(F.softmax(torch.stack([pos_sim, neg_sim]) / temperature, dim=0))
    return loss.mean()

class RGBLoss(nn.Module):
    def __init__(self):
        super(RGBLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        # Ensure input and target are in the same range and data type
        input = input.float()
        target = target.float()
        
        # Compute RGB loss
        rgb_loss = self.mse_loss(input, target)
        return rgb_loss


class MySobelxy(nn.Module):
    def __init__(self):
        super(MySobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)
        # Apply convolution for all channels in the batch
        sobelx = F.conv2d(x, self.weightx, padding=1, groups=x.shape[1])
        sobely = F.conv2d(x, self.weighty, padding=1, groups=x.shape[1])
        
        # Calculate the gradient magnitude
        grad = torch.abs(sobelx) + torch.abs(sobely)
        
        return grad


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
        eps
        + torch.sqrt(torch.sum(img1**2, dim=-1))
        * torch.sqrt(torch.sum(img2**2, dim=-1))
    )
    cc = torch.clamp(cc, -1.0, 1.0)
    return cc.mean()
