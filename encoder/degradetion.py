import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import os


class DegradationModel(nn.Module):
    def __init__(self, max_mask_num=5):
        super(DegradationModel, self).__init__()

        # first order
        self.brightness_factor_1 = nn.Parameter(torch.tensor(0.9))  
        self.contrast_factor_1 = nn.Parameter(torch.tensor(0.9)) 
        self.noise_std_1 = nn.Parameter(torch.tensor(0.01)) 
        self.jpeg_quality_1 = nn.Parameter(torch.tensor(50.0))

        # second order
        self.brightness_factor_2 = nn.Parameter(torch.tensor(0.9))  
        self.contrast_factor_2 = nn.Parameter(torch.tensor(0.9))  
        self.noise_std_2 = nn.Parameter(torch.tensor(0.03))  
        self.jpeg_quality_2 = nn.Parameter(torch.tensor(30.0))
        
        # blur kernel
        kernel = torch.tensor([[1,  4,  6,  4,  1],
                                [4, 16, 24, 16,  4],
                                [6, 24, 36, 24,  6],
                                [4, 16, 24, 16,  4],
                                [1,  4,  6,  4,  1]], dtype=torch.float32)
        # kernel *= 6
        self.blur_kernel = nn.Parameter(kernel.unsqueeze(0).unsqueeze(0) / torch.sum(kernel))
        
        # self.parameters = [self.brightness_factor_1, self.brightness_factor_2,
        #                   self.contrast_factor_1, self.contrast_factor_2,
        #                   self.noise_std_1, self.noise_std_2,
        #                   self.jpeg_quality_1, self.jpeg_quality_2,
        #                   self.blur_kernel]
        # upsample or downsample
        self.sample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=0.5, mode='nearest'),
            nn.Upsample(scale_factor=0.5, mode='bicubic', align_corners=False),
        ])
        
        # max number of mask
        self.max_mask_num = max_mask_num  

    def create_motion_blur_kernel(self, degree=15, angle=45):
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        return motion_blur_kernel

    def sharpen(self, image):
        _, c, _, _ = image.shape
        kernel = torch.tensor([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
        sharp_image = F.conv2d(image, kernel, padding=1, groups=c)
        return sharp_image

    def blur(self, image, size=25):
        _, c, _, _ = image.shape
        # angle = random.uniform(-np.pi/6, np.pi/6)  # 随机角度
        angle = random.randint(-45, 45)
        degree = random.randint(9, 15)
        degree = degree + 1 if degree % 2 == 0 else degree

        # motion kernel
        kernel = self.create_motion_blur_kernel(angle=angle, degree=degree)
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        
        # blur_image = F.conv2d(image, self.blur_kernel.repeat(c, 1, 1, 1), padding=2, groups=c)
        blur_image = F.conv2d(image, kernel.repeat(c, 1, 1, 1), padding=degree // 2, groups=c)
        blur_image = (blur_image - torch.min(blur_image)) / (torch.max(blur_image) - torch.min(blur_image))
        return blur_image

    def brightness_contrast_degradation(self, image, brightness_factor, contrast_factor, infrared=False):
        _, c, _, _ = image.shape
        # there is no need to adjust the image brightness for IR images
        if infrared == 3:
            degraded_image = image * brightness_factor
        else:
            degraded_image = image
        mean = degraded_image.mean()
        degraded_image = (degraded_image - mean) * contrast_factor + mean
        return degraded_image

    def add_noise(self, image, noise_std):
        noise = torch.randn_like(image) * noise_std
        noisy_image = image + noise
        return noisy_image.clamp(0, 1)

    def jpeg_compress(self, image, quality):
        batch_size, channels, height, width = image.shape
        image_jpeg = [] 
        for i in range(batch_size):
            # print(image[i].shape)
            numpy_img = image[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality.item())]
            _, encimg = cv2.imencode('.jpg', numpy_img, encode_param)
            decimg = cv2.imdecode(encimg, 1) / 255.0
            if channels == 1:
                decimg = decimg[:, :, 0:1]
                # decimg = np.expand_dims(decimg, axis=-1)

            compressed_image = torch.tensor(decimg).permute(2, 0, 1).to(image.device)
            image_jpeg.append(compressed_image)
        
        # 将所有压缩后的图像重新拼接成一个 batch
        image_jpeg = torch.stack(image_jpeg)
        # print(image_jpeg.shape)
        return image_jpeg

    def create_square_mask(self, image, num_masks):
        """Generate multiple square random masks, each with random size and position"""
        _, c, h, w = image.size()

        # create all zero mask
        mask = torch.zeros((1, h, w), device=image.device)
        for _ in range(num_masks):
            # randomly generate square sizes
            square_size = torch.randint(80, min(h, w) // 3, (1,)).item()

            # randomly select the coordinates of the upper left corner of the square
            top_left_x = torch.randint(0, w - square_size + 1, (1,)).item()
            top_left_y = torch.randint(0, h - square_size + 1, (1,)).item()

            # set the area of square is 1
            mask[:, top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = 1
        return mask

    def forward(self, image, infrared=False):
        # set a random number of masks, the maximum number is self.max_mask_num
        num_masks = torch.randint(1, self.max_mask_num + 1, (1,)).item()  # 随机选择掩码数量
        mask = self.create_square_mask(image, num_masks)

        sample_id = random.choice([0, 1, 2, 3, 4, 5])
        # first order
        # sharpe = self.sharpen(image)
        blur1 = self.blur(image)
        # print("blur1:", blur1.shape)
        # bright_contrast1 = self.brightness_contrast_degradation(blur1, self.brightness_factor_1, self.contrast_factor_1, infrared)
        # print("bright_contrast1:", bright_contrast1.shape)
        sample1 = self.sample[sample_id](blur1)#bright_contrast1)
        image = self.add_noise(image, self.noise_std_1)
        noise1 = self.add_noise(sample1, self.noise_std_1)
        # print("noise1:", noise1.shape)
        # jpeg1 = self.jpeg_compress(sample1, self.jpeg_quality_1)
        # print("jpeg1:", jpeg1.shape)

        # second order
        # blur2 = self.blur(noise1)#jpeg1.float()) 
        # bright_contrast2 = self.brightness_contrast_degradation(blur2, self.brightness_factor_2, self.contrast_factor_2)
        sample2 = self.sample[5 - sample_id](noise1)#bright_contrast2)
        image = self.add_noise(image, self.noise_std_2)
        # jpeg2 = self.jpeg_compress(sample2, self.jpeg_quality_2)

        # use mask to fuse the degraded area with the original image
        # final_image = jpeg2 * mask + image * (1 - mask)
        final_image = sample2 * mask + image * (1 - mask)
        return final_image.float()

def random_swap(rgb, modal_x, beta=0.8):
    H, W, C = rgb.shape
    lam = np.random.beta(beta, beta)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    patch_rgb = rgb[bby1:bby2, bbx1:bbx2, :].copy()
    patch_modal_x = modal_x[bby1:bby2, bbx1:bbx2, :].copy()

    rgb[bby1:bby2, bbx1:bbx2, :] = patch_modal_x
    modal_x[bby1:bby2, bbx1:bbx2, :] = patch_rgb
    return rgb, modal_x


model = DegradationModel().cuda()
def convert_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cuda()# / 255.  # HWC -> CHW，并增加batch维度
            with torch.no_grad():
                output_tensor = model(image_tensor)
            # output_tensor = (output_tensor - torch.min(output_tensor)) / (torch.max(output_tensor) - torch.min(output_tensor))
            output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  
            output_image = (output_image).astype(np.uint8)  
            output_image_path = os.path.join(output_folder, filename)
            print(filename)
            cv2.imwrite(output_image_path, output_image)

if __name__ == "__main__":
    # import torchvision
    vi = cv2.imread('00093D.png') / 255.
    ir = cv2.imread('/home/suguilin/myfusion/datasets/MFNet/Modal/00093D.png') / 255.
    # vi, ir = random_swap(vi, ir)
    # # vi = cv2.imread('00093D.png', 0) / 255.
    vi = torch.from_numpy(vi).float()
    ir = torch.from_numpy(ir).float()
    # cv2.imwrite('vi_inputD.jpg', vi.cpu().numpy() * 255)  
    vi = vi.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    ir = ir.unsqueeze(0).permute(0, 3, 1, 2).cuda()
    # R_vis = torchvision.transforms.functional.adjust_gamma(vi, 0.7, 1)
    # R_ir = torchvision.transforms.functional.adjust_contrast(ir, 1.2)
    # cv2.imwrite('vi_inputD_gamma.jpg', R_vis.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255)  
    # cv2.imwrite('vi_inputD_contrast.jpg', R_ir.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255)
    # # vi = vi.unsqueeze(0).unsqueeze(0).cuda()
    # print(vi.shape)
    degradation_model = DegradationModel().cuda()
    output_image = degradation_model(vi)
    # # for i in range(len(para)):
    # #     print(para[i])
    # print(output_image.shape)
    cv2.imwrite('outputD.png', output_image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)  
    # input_folder = '/home/suguilin/Graduation/myfusion/datasets/MFNet/Modal' 
    # output_folder = '/home/suguilin/MSRS_Deg/ir_blur'  
    # convert_images_in_folder(input_folder, output_folder)

    # import numpy as np
    # import cv2
    # def motion_blur(image, degree=15, angle=45):
    #     image = np.array(image)
    #     #这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    #     M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    #     motion_blur_kernel = np.diag(np.ones(degree))
    #     motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    #     motion_blur_kernel = motion_blur_kernel / degree
    #     blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    #     #convert to uint8
    #     cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    #     blurred = np.array(blurred, dtype=np.uint8)
    #     return blurred
    # img = cv2.imread('00093D.png')
    # #运动模糊
    # img_motion = motion_blur(img)
    # #高斯模糊
    # img_gauss = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=0, sigmaY=0)
    # cv2.imwrite("motion_blur" + ".jpg",img_motion )
    # cv2.imwrite("GaussianBlur" + ".jpg",img_gauss )


