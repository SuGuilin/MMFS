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
from utils.engine import ensure_dir, parse_devices
from utils.evaluator import Evaluator, FuseEvaluator
from utils.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from model.MMoEFusion import MMoEFusion
from dataloader.dataloader import ValPre
from utils.print_indicators import Print_Indicators

logger = get_logger()

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
        # link_file(log_file, log_file_link)

        for model in models:
            logger.info("Load Model: %s" % model)
            self.val_func = load_model(self.network, model)
            if len(self.devices) == 1:
                result_line, mean_metric, metric_result = self.single_process_evalutation()
            else:
                result_line, mean_metric, metric_result = self.single_process_evalutation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            results.flush()

        results.close()
        return metric_result, mean_metric
    
    def func_per_iteration(self, data, device, config):
        img_rgb = data['data']
        img_ir = data['modal_x']
        name = data['fn']
        # print(img_rgb.shape)
        # print(img_rgb)
        # print(img_ir.shape)
        # print(img_ir)
        _, img_rgb_Cr, img_rgb_Cb = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb))
        img_rgb = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YCrCb))[0][np.newaxis, ...] / 255.0 # (1, 480, 640)
        img_ir = np.expand_dims(img_ir, axis=0)  # (1, 480, 640)

        img_rgb = torch.from_numpy(img_rgb).to(device).float().unsqueeze(0)
        img_ir = torch.from_numpy(img_ir).to(device).float().unsqueeze(0)

        with torch.no_grad():
            self.val_func = self.network.to(device)
            output_fused = self.val_func(img_rgb, img_ir)
            output_fused = (output_fused - torch.min(output_fused)) / (torch.max(output_fused) - torch.min(output_fused))
        
        results_dict = {'output_fused': np.squeeze(output_fused * 255).cpu().numpy().astype(np.uint8)}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn = name + '.png'
            output_fused = output_fused * 255
            fusion_img = output_fused.squeeze().cpu().numpy().astype(np.uint8)
            cv2.imwrite('test.png', fusion_img)
            ycrcb_fusion_img = np.dstack((fusion_img, img_rgb_Cr, img_rgb_Cb))
            rgb_fusion_img = cv2.cvtColor(ycrcb_fusion_img, cv2.COLOR_YCrCb2BGR)
            # rgb_fusion_img = cv2.cvtColor(rgb_fusion_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.save_path, fn), rgb_fusion_img)
            logger.info('Save the image ' + fn)

        return results_dict
    
    def single_process_evalutation(self):
        start_eval_time = time.perf_counter()
        logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []
        # 记录指标值
        metric_result = np.zeros((8))

        for idx in tqdm(range(self.ndata)):
            item = self.dataset[idx]  # dict(data=rgb, modal_x=x, fn=str(item_name), n=len(self._file_names))
            # 2024-06-10 20:01 修改输出指标
            
            # ir = cv2.split(item['modal_x'])[0]
            # print(item['data'])
            # print(item['modal_x'])
            vi = (item['data']).astype(np.uint8)
            ir = (item['modal_x'] * 255).astype(np.uint8)
            vi = cv2.cvtColor(vi, cv2.COLOR_BGR2GRAY)

            results_dict = self.func_per_iteration(item, self.devices[0], self.config)
            fi = results_dict['output_fused']
            # fi = np.transpose(results_dict['output_fused'], (1, 2, 0))
            # fi = cv2.split(cv2.cvtColor(fi, cv2.COLOR_BGR2YCrCb))[0]
            
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

            metric_result += metrics
            all_results.append(results_dict)

        metric_result /= len(all_results)
        
        result_line, mean_metric = self.compute_metric(metric_result, self.config)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mean_metric, metric_result
    
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
    parser.add_argument('-e', '--epochs', default='435', type=str) #'last'
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    # engine = Engine(custom_parser=parser, config_path='./configs/config.yaml')
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        config = load_config('./configs/config.yaml')
    elif dataset_name == 'm3fd':
        config = load_config('./configs/config_m3fd.yaml')
    elif dataset_name == 'roadscene':
        config = load_config('./configs/config_roadscene.yaml')
    elif dataset_name == 'tno':
        config = load_config('./configs/config_tno.yaml')
    else:
        raise ValueError('Not a valid dataset name')

    network = MMoEFusion()   # 以后可以进一步使用config来设置模型的各个参数
    model_name = 'MMoEFusion'

    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre, testflag=True)

    with torch.no_grad():
        fuser = TestEvaluator(dataset=dataset, save_path='./test/results1',
                                    network=network, devices=all_dev, verbose=args.verbose,
                                    config=config)
        metric_result, mean_metric = fuser.run(config.checkpoint_dir, args.epochs, config.test_log_file)
        Print_Indicators(model_name, metric_result)