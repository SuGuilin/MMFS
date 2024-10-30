import cv2
import torch
import numpy as np
from torch.utils import data
import random
from skimage.restoration import denoise_tv_chambolle
from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)


def random_mirror(rgb, modal_x, label, guide):
    # if random.random() >= 0.5:
    #     rgb = cv2.flip(rgb, 1)
    #     modal_x = cv2.flip(modal_x, 1)
    flip_code = np.random.choice([-1, 0, 1])  # -1:水平垂直翻转, 0:垂直翻转, 1:水平翻转
    rgb = cv2.flip(rgb, flip_code)
    modal_x = cv2.flip(modal_x, flip_code)
    label = cv2.flip(label, flip_code)
    guide = cv2.flip(guide, flip_code)
    return rgb, modal_x, label, guide


def random_scale(rgb, modal_x, label, guide, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (sw, sh), interpolation=cv2.INTER_NEAREST)
    guide = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return rgb, modal_x, label, guide, scale

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


def gaussian_blur(image, kernel_size=5, sigma=1):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image

# Gaussian Blur
def denoise_image(rgb, modal_x):
    rgb = cv2.GaussianBlur(rgb, (5, 5), 1)
    modal_x = cv2.GaussianBlur(modal_x, (5, 5), 1)
    return rgb, modal_x

# Bilateral filtering, keep edges and smooth the image
def denoise_image_bilateral(rgb, modal_x):
    rgb = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    modal_x = cv2.bilateralFilter(modal_x, d=9, sigmaColor=75, sigmaSpace=75)
    return rgb, modal_x

# Non-local denoising, average denoising of similar blocks in the image to maintain image details
def denoise_image_nl_means(rgb, modal_x):
    rgb = cv2.fastNlMeansDenoisingColored(rgb, None, h=10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
    modal_x = cv2.fastNlMeansDenoising(modal_x, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return rgb, modal_x

# Median filter, remove salt and pepper noise, keep the edge
def denoise_image_median(rgb, modal_x):
    rgb = cv2.medianBlur(rgb, ksize=5)
    modal_x = cv2.medianBlur(modal_x, ksize=5)
    return rgb, modal_x

# Total variation denoising, remove noise and preserve image details
def denoise_image_tv(rgb, modal_x):
    rgb = denoise_tv_chambolle(rgb, weight=0.1)
    modal_x = denoise_tv_chambolle(modal_x, weight=0.1)
    return rgb, modal_x

class TrainPre(object):
    def __init__(self, norm_mean, norm_std, config):
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.config = config

    def __call__(self, rgb, modal_x, label, guide):
        # rgb, modal_x = denoise_image(rgb, modal_x)
        # rgb, modal_x = denoise_image_tv(rgb, modal_x)
        rgb, modal_x, label, guide = random_mirror(rgb.transpose(1, 2, 0), modal_x, label, guide)
        # rgb, modal_x = random_swap(rgb, modal_x)
        # rgb = rgb.transpose(2, 0, 1) 
        # print("rgb:", rgb.shape)
        # print("modal_x:", modal_x.shape) 
        # 不需要随机裁剪就取消了
        # Randomly scale and then crop to a fixed size
        if self.config.train_scale_array is not None:
            rgb, modal_x, label, guide, scale = random_scale(
                rgb, modal_x, label, guide, self.config.train_scale_array
            )
        # rgb = normalize(rgb, self.norm_mean, self.norm_std)
        # modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)
        p_guide, _ = random_crop_pad_to_shape(guide, crop_pos, crop_size, 0)
        p_label, _ = random_crop_pad_to_shape(label, crop_pos, crop_size, 0)

        # p_rgb = cv2.resize(p_rgb, (384, 288))
        # p_modal_x = cv2.resize(p_modal_x, (384, 288))
        # p_label = cv2.resize(p_label, (384, 288)) 
        # # p_guide = guide
        # p_guide = cv2.resize(p_guide, (384, 288)) # guide

        # p_rgb = np.expand_dims(p_rgb, axis=0)
        p_rgb = p_rgb.transpose(2, 0, 1) 
        # p_modal_x = np.expand_dims(p_modal_x, axis=0)
        p_modal_x = p_modal_x.transpose(2, 0, 1) 
        # p_guide = np.expand_dims(p_guide, axis=0)
        p_guide = p_guide.transpose(2, 0, 1)
        # p_rgb = p_rgb.transpose(2, 0, 1)
        # p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_modal_x, p_label, p_guide
        # return rgb, modal_x


class ValPre(object):
    def __call__(self, rgb, modal_x, label):
        # rgb = cv2.resize(rgb.transpose(1, 2, 0), (384, 288)).transpose(2, 0, 1)
        # modal_x = cv2.resize(modal_x, (384, 288))
        # label = cv2.resize(label, (384, 288))
        return rgb, modal_x, label

# 如果直接读入文本可将这注释取消
def custom_collate_fn(batch):
    rgb = [item['data'] for item in batch]
    modal_x = [item['modal_x'] for item in batch]
    
    des_rgb = [item['des_rgb'] for item in batch]
    des_x = [item['des_x'] for item in batch]
    guide = [item['guide'] for item in batch]
    label = [item['label'] for item in batch]

    return {
        'data': torch.stack(rgb), 
        'modal_x': torch.stack(modal_x),  
        'des_rgb': torch.stack(des_rgb),  
        'des_x': torch.stack(des_x),    
        'guide': torch.stack(guide),
        'label': torch.stack(label),
        'fn': [item['fn'] for item in batch],
        'n': batch[0]['n'],  
    }

def get_train_loader(engine, dataset, config):
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "text_root": config.text_root_folder,
        "text_format": config.text_format,
        "guide_root": config.guide_root_folder,
        "guide_format": config.guide_format,
        "label_root": config.label_root_folder,
        "label_format": config.label_format,
        "class_names": config.class_names,
        "x_single_channel": config.x_is_single_channel,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std, config)

    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
        False,
        True,
        True
    )
    #assert isinstance(train_dataset, torch.utils.data.Dataset), \
    #    f"train_dataset 应该是一个 torch.utils.data.Dataset 实例, 实际类型: {type(train_dataset)}"

    # 打印 train_dataset 的类型和长度以进行调试
    # print(f"train_dataset 类型: {type(train_dataset)}")
    # print(f"train_dataset 内容: {train_dataset.__dict__}")
    # print(f"train_dataset 长度: {len(train_dataset)}")

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=True,
        shuffle=is_shuffle,
        pin_memory=True,
        sampler=train_sampler,
    )

    return train_loader, train_sampler
