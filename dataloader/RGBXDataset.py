import os
from pickletools import uint8
import cv2
import torch
import numpy as np
import torch.utils.data as data
import logging

logging.basicConfig(filename='data_loading.log', level=logging.INFO)


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None, testflag=False, guideflag=False, labelflag=False):
        super(RGBXDataset, self).__init__()
        self._split_name = split_name
        self._rgb_path = setting['rgb_root']
        self._rgb_format = setting['rgb_format']
        self._x_path = setting['x_root']
        self._x_format = setting['x_format']
        self._text_path = setting['text_root']
        self._text_format = setting['text_format']
        self._guide_path = setting['guide_root']
        self._guide_format = setting['guide_format']
        self._label_path = setting['label_root']
        self._label_format = setting['label_format']
        self.class_names = setting['class_names']
        self._x_single_channel = setting['x_single_channel']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.testflag = testflag
        self.guideflag = guideflag
        self.labelflag = labelflag

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]
        # print(item_name)
        # logging.info(f"Loading item: {item_name}")
        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path = os.path.join(self._x_path, item_name + self._x_format)
        rgb_text_path = os.path.join(self._text_path, 'RGB', item_name + self._text_format)
        x_text_path = os.path.join(self._text_path, 'Modal', item_name + self._text_format)
        guide_path = os.path.join(self._guide_path, item_name + self._guide_format)
        label_path = os.path.join(self._label_path, item_name + self._label_format)

        # with open(rgb_text_path, 'r', encoding='utf-8') as f1:
        #     des_rgb = f1.readlines()
        #     # print(f"RGB Description: {des_rgb}")
        # with open(x_text_path, 'r', encoding='utf-8') as f2:
        #     des_x = f2.readlines()
        #     # print(f"X Description: {des_x}")
        des_rgb = np.load(rgb_text_path)[0]
        des_x = np.load(x_text_path)[0]

        # Check the following settings if necessary
        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB, np.float32, self.testflag)
        # rgb = self._open_image(rgb_path, cv2.IMREAD_GRAYSCALE)
        # guide = self._open_image(guide_path, cv2.IMREAD_GRAYSCALE, np.float32, guideflag=self.guideflag)
        guide = self._open_image(guide_path, cv2.COLOR_BGR2RGB, np.float32, guideflag=self.guideflag)
        label = self._open_image(label_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8, labelflag=self.labelflag)
        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE, np.float32)
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, cv2.COLOR_BGR2RGB, np.float32)
        
        if self.preprocess is not None and self._split_name == "train": 
            rgb, x, label, guide = self.preprocess(rgb, x, label, guide)
        if self.preprocess is not None and self._split_name == "val": 
            rgb, x, label = self.preprocess(rgb, x, label)

        if self._split_name == "train":
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
            des_rgb = torch.from_numpy(np.ascontiguousarray(des_rgb)).float()
            des_x = torch.from_numpy(np.ascontiguousarray(des_x)).float()
            guide = torch.from_numpy(np.ascontiguousarray(guide)).float()
            label = torch.from_numpy(np.ascontiguousarray(label)).long()

        output_dict = dict(
            data=rgb, modal_x=x, des_rgb=des_rgb, des_x=des_x, guide=guide, label=label, fn=str(item_name), n=len(self._file_names)
        )

        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ["train", "val"]
        source = self._train_source
        if split_name == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            file_names.append(file_name)

        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)
        rand_indices = torch.randperm(files_len).tolist()

        # 取出索引列表中前 length % files_len 个索引，以用于补充文件列表
        new_indices = rand_indices[: length % files_len]
        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None, testflag=False, guideflag=False, labelflag=False):
        if testflag is True:
            img = cv2.imread(filepath).astype(dtype)
        elif guideflag is True:
            # img = cv2.imread(filepath, mode)[None, :, :].astype(dtype) / 255.
            img = cv2.imread(filepath, mode).astype(dtype) / 255.
        elif labelflag is True:
            img = cv2.imread(filepath, mode)[None, :, :].astype(dtype)
        elif mode == cv2.COLOR_BGR2RGB:
            img = cv2.imread(filepath, mode).transpose(2, 0, 1).astype(dtype) / 255.  
            # print(img.shape)
            # print(img.min(), img.max())
            # cv2.imwrite('x_rgb.png', img.astype(np.uint8).transpose(1, 2, 0))
            # img = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000 # 由于文本描述是可见光和红外分开的，这里先读入三通道的可见光，若读入Y通道删除注释即可
            # print(img.shape)
        else:
            img = cv2.imread(filepath, mode)[None, :, :].astype(dtype) / 255.  # 
            # print(img.min(), img.max())
            # cv2.imwrite('x_ir.png', img.transpose(1, 2, 0).astype(np.uint8))
            # img = np.expand_dims(img, axis=0)
            # print(img.shape)
            # img = np.squeeze(img, axis=0)
        if not testflag and mode != cv2.COLOR_BGR2RGB:
            img = np.squeeze(img, axis=0)
        return img
    
    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

