import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import yaml
import time
from tqdm import tqdm
from easydict import EasyDict as edict
from collections import OrderedDict
from timm.models.layers import to_2tuple
from utils.engine import ensure_dir, parse_devices
from utils.evaluator import Evaluator, FuseEvaluator
from utils.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from model.MMoEFusion import MMoEFusion
from dataloader.dataloader import ValPre

from utils.print_indicators import Print_Indicators
from utils.transforms import pad_image_to_shape, normalize

logger = get_logger()

def get_class_colors():
    pattale = [
        [0, 0, 0],        # unlabelled
        [128, 0, 64],     # car
        [0, 64, 64],      # person
        [192, 128, 0],    # bike
        [192, 0, 0],      # curve
        [0, 128, 128],    # car_stop
        [128, 64, 64],    # guardrail
        [128, 128, 192],  # color_cone
        [0, 64, 192],     # bump
    ]
    return pattale

class TestEvaluator(Evaluator):
    def run(self, model_path, model_indice, log_file): #, log_file_link):
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "_" in model_indice:
            start_epoch = int(model_indice.split("-")[0])
            end_epoch = model_indice.split("-")[1]
            models = os.listdir(model_path)
            models.remove("epoch-last.pth")
            sorted_models = [None] * len(models)
            model_idx = [0] * len(models)

            for idx, m in enumerate(models):
                num = m.split(".")[0].split("-")[1]
                model_idx[idx] = num
                sorted_models[idx] = m
            model_idx = np.array([int(i) for i in model_idx])

            down_bound = model_idx >= start_epoch
            up_bound = [True] * len(sorted_models)
            if end_epoch:
                end_epoch = int(end_epoch)
                assert start_epoch < end_epoch
                up_bound = model_idx <= end_epoch
            bound = up_bound * down_bound
            model_slice = np.array(sorted_models)[bound]
            models = [os.path.join(model_path, model) for model in model_slice]
        else:
            if os.path.exists(model_path):
                models = [os.path.join(model_path, 'epoch-%s.pth' % model_indice), ]
            else:
                models = [None]

        results = open(log_file, 'a')

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            if len(self.devices) == 1:
                result_line, mSSIM, metric_result, seg_lines, mIoU, mean_pixel_acc = self.single_process_evalutation()
            else:
                result_line, mSSIM, metric_result, seg_lines, mIoU, mean_pixel_acc = self.single_process_evalutation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            for line in seg_lines:
                results.write(line + '\n') 
            results.flush()
        results.close()
        seg_key = mIoU + mean_pixel_acc + mSSIM
        return metric_result, seg_key, mSSIM, mIoU, mean_pixel_acc


    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()
        logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []

        # record segmentation results
        hist = np.zeros((self.class_num, self.class_num))
        correct = 0
        labeled = 0

        # record fusion results
        fusion_metric_result = np.zeros((8))

        for idx in tqdm(range(self.ndata)):
            item = self.dataset[idx]  # dict(data=rgb, modal_x=x, fn=str(item_name), n=len(self._file_names))
            
            ir = (item['modal_x'] * 255).transpose(2, 0, 1)[0, :, :].astype(np.uint8)
            vi = cv2.cvtColor((item['data'] * 255).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

            results_dict = self.func_per_iteration(item, self.devices[0], self.config)
            fi = results_dict['output_fused']
            fi = np.transpose(fi, (1, 2, 0))
            fi = cv2.cvtColor(fi, cv2.COLOR_RGB2GRAY)
            
            hist += results_dict['hist']
            correct += results_dict['correct']
            labeled += results_dict['labeled']

            metrics = np.array([
                FuseEvaluator.EN(fi),
                FuseEvaluator.SD(fi),
                FuseEvaluator.SF(fi),
                FuseEvaluator.MI(fi, ir, vi),
                FuseEvaluator.SCD(fi, ir, vi),
                FuseEvaluator.VIFF(fi, ir, vi),
                FuseEvaluator.Qabf(fi, ir, vi), 
                FuseEvaluator.SSIM(fi, ir, vi)
                ]
            )

            fusion_metric_result += metrics
            all_results.append(results_dict)

        fusion_metric_result /= len(all_results)
        
        result_line, mSSIM = self.compute_metric(fusion_metric_result, self.config)
        seg_lines, mIoU, mean_pixel_acc = self.compute_score(hist, correct, labeled)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mSSIM, fusion_metric_result, seg_lines, mIoU, mean_pixel_acc
    
    def func_per_iteration(self, data, device, config):
        img_rgb = data['data']
        label = data['label']
        img_ir = data['modal_x']
        text_rgb = data['des_rgb']
        text_ir = data['des_x']
        name = data['fn']

        # import pdb; pdb.set_trace()
        pred = self.sliding_eval_rgbX((img_rgb*255).transpose(1, 2, 0), (img_ir*255), self.eval_crop_size, self.eval_stride_rate, device)
        
        img_rgb = torch.from_numpy(img_rgb).to(device).float().unsqueeze(0)
        img_ir = torch.from_numpy(img_ir).to(device).float().unsqueeze(0).permute(0, 3, 1, 2)
        text_rgb = torch.from_numpy(text_rgb).to(device).float().unsqueeze(0)
        text_ir = torch.from_numpy(text_ir).to(device).float().unsqueeze(0)

        with torch.no_grad():
            self.val_func = self.network.to(device)
            output_fused, output_semantic, _ = self.val_func(img_rgb, img_ir, text_rgb, text_ir)
            output_fused = (output_fused - torch.min(output_fused)) / (torch.max(output_fused) - torch.min(output_fused))

        # output_semantic = torch.exp(output_semantic[0]).permute(1, 2, 0)
        # pred = output_semantic.squeeze().cpu().numpy().argmax(2)
        hist_tmp, labeled_tmp, correct_tmp = self.hist_info(self.class_num, pred, label)
        results_dict = {
            'output_fused': np.squeeze(output_fused * 255).cpu().numpy().astype(np.uint8),
            'hist': hist_tmp, 
            'labeled': labeled_tmp, 
            'correct': correct_tmp,
        }

        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn = name + '.png'
            fn_fus = name + '_fus.png'
            fn_seg = name + '_seg.png'
            fn_lab = name + '_lab.png'
            fn_com = name + '_com.png'

            output_fused = output_fused * 255
            fusion_img = output_fused.squeeze().cpu().numpy().astype(np.uint8)
            fusion_img = fusion_img.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(self.save_path, fn_fus), fusion_img)
            # ycrcb_fusion_img = np.dstack((fusion_img, img_rgb_Cr, img_rgb_Cb))
            # rgb_fusion_img = cv2.cvtColor(ycrcb_fusion_img, cv2.COLOR_YCrCb2BGR)
            # rgb_fusion_img = cv2.cvtColor(rgb_fusion_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(self.save_path, fn), rgb_fusion_img)

            class_colors = config.pattale #get_class_colors()
            segment_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for i in range(config.num_classes):
                segment_img[pred == i] = class_colors[i]

            for i in range(self.class_num):
                label_img[label == i] = class_colors[i]

            combined_img = cv2.hconcat([fusion_img, segment_img, label_img])
            cv2.imwrite(os.path.join(self.save_path, fn_com), combined_img)

            logger.info('Save the image ' + fn)

        return results_dict


    def sliding_eval_rgbX(self, img, modal_x, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales: # self.multi_scales = [1.0]
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            if len(modal_x.shape) == 2:
                modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            else:
                modal_x_scale = cv2.resize(modal_x, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process_rgbX(img_scale, modal_x_scale, (ori_rows, ori_cols), crop_size, stride_rate, device)
        pred = processed_pred.argmax(2)
        return pred


    def scale_process_rgbX(self, img, modal_x, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if new_cols <= crop_size[1] or new_rows <= crop_size[0]: # Enter the interface

            # img: (480, 640, 3)  modal_x: (480, 640, 3)
            input_data, input_modal_x, margin = self.process_image_rgbX(img, modal_x, crop_size)
            score = self.val_func_process_rgbX(input_data, input_modal_x, device) 
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            modal_x_pad, margin = pad_image_to_shape(modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0]
                    s_y = grid_yidx * stride[1]
                    e_x = min(s_x + crop_size[0], pad_cols)
                    e_y = min(s_y + crop_size[1], pad_rows)
                    s_x = e_x - crop_size[0]
                    s_y = e_y - crop_size[1]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    if len(modal_x_pad.shape) == 2:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x]
                    else:
                        modal_x_sub = modal_x_pad[s_y:e_y, s_x: e_x,:]

                    # make input meet needs
                    input_data, input_modal_x, tmargin = self.process_image_rgbX(img_sub, modal_x_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, input_modal_x, device)
                    
                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]), tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output
    
    def val_func_process_rgbX(self, input_data, input_modal_x, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
    
        input_modal_x = np.ascontiguousarray(input_modal_x[None, :, :, :], dtype=np.float32)
        input_modal_x = torch.FloatTensor(input_modal_x).cuda(device)
        text_ir = torch.randn(1, 128, 128)
        text_rgb = torch.randn(1, 128, 128)
        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                Fus_img, score, _ = self.val_func(input_data, input_modal_x, text_rgb, text_ir)
                score = score[0]
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_modal_x = input_modal_x.flip(-1)
                    score_flip, Fus_img_flip = self.val_func(input_data, input_modal_x)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)
        
        return score

    # for rgbd segmentation
    def process_image_rgbX(self, img, modal_x, crop_size=None):
        p_img = img
        p_modal_x = modal_x
    
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)
    
        p_img = normalize(p_img)          # / 255.0
        p_modal_x = normalize(p_modal_x)  # / 255.0
    
        if crop_size is not None:  # Enter the interface
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            if len(modal_x.shape) == 2:
                p_modal_x = p_modal_x[np.newaxis, ...]
            else:
                p_modal_x = p_modal_x.transpose(2, 0, 1) # 3 H W
            
            # compared with img and modal_x, p_img, p_modal_x only div 255.0 and transpose
            return p_img, p_modal_x, margin
    
        p_img = p_img.transpose(2, 0, 1) # 3 H W

        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)
        
        return p_img, p_modal_x


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return edict(config)

def load_model(model, model_file, is_restore=False):
    t_start = time.time()

    if model_file is None:
        return model

    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location='cpu')
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        elif 'module' in state_dict.keys():
            state_dict = state_dict['module']
    else:
        state_dict = model_file
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='210', type=str) #'last'
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    # engine = Engine(custom_parser=parser, config_path='./configs/config.yaml')
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        config = load_config('./configs/config_mfnet.yaml')
    elif dataset_name == 'fmb':
        config = load_config('./configs/config_fmb.yaml')
    elif dataset_name == 'roadscene':
        config = load_config('./configs/config_roadscene.yaml')
    elif dataset_name == 'tno':
        config = load_config('./configs/config_tno.yaml')
    else:
        raise ValueError('Not a valid dataset name')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 
    network = MMoEFusion(
        device=device, 
        num_classes=config.num_classes, 
        embed_dim=config.decoder_embed_dim, 
        align_corners=config.align_corners,
    )   # 以后可以进一步使用config来设置模型的各个参数
    model_name = 'MMoEFusion'
    network.eval() 

    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    "text_root": config.text_root_folder,
                    "text_format": config.text_format,
                    "guide_root": config.guide_root_folder,
                    "guide_format": config.guide_format,
                    "label_root": config.label_root_folder,
                    "label_format": config.label_format,
                    "class_names": config.class_names,
                    'x_single_channel': config.x_is_single_channel,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,}
    
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre, labelflag=True)

    with torch.no_grad():
        fuser = TestEvaluator(dataset=dataset, save_path='./test/results4',
                                    network=network, devices=all_dev, verbose=args.verbose,
                                    config=config)
        
        metric_result, mean_metric, mSSIM, mIoU, mean_pixel_acc = fuser.run(config.checkpoint_dir, args.epochs, config.test_log_file)
        Print_Indicators(model_name, metric_result)