import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
from skimage.metrics import structural_similarity as ssim
import torch
import multiprocessing as mp

from utils.logger import get_logger
from utils.engine import link_file, ensure_dir
from utils.transforms import pad_image_to_shape, normalize
from PIL import Image

logger = get_logger()

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

class FuseEvaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None): 
        if imgA is None:
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        h = h[h > 0]  # 过滤掉0概率
        return -np.sum(h * np.log2(h))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return skm.mutual_info_score(image_F.flatten(), image_A.flatten()) + skm.mutual_info_score(image_F.flatten(), image_B.flatten())

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B): # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls,ref, dist): # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls,img):
        # Sobel Operator Sobel
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F, image_A, data_range=image_F.max() - image_F.min()) + ssim(image_F,image_B, data_range=image_F.max() - image_F.min())

def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        # Create a Gaussian kernel as MATLAB's
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB


class Evaluator(object):
    def __init__(self, dataset, network, devices, verbose=False, save_path=None, config=None):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length()
        self.network = network
        self.devices = devices
        self.eval_crop_size = config.eval_crop_size
        self.eval_stride_rate = config.eval_stride_rate
        self.class_num = config.num_classes
        self.multi_scales = config.eval_scale_array
        self.is_flip = config.eval_flip

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.save_label = False
        self.results_queue = self.context.Queue(self.ndata)
        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.config = config

    def run(self, model_path, model_indice, log_file): #, log_file_link):
        if '.pth' in model_indice:
            models = [model_indice, ]
        elif "-" in model_indice:
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
            self.val_func = self.network # load_model(self.network, model)
            if len(self.devices) == 1:
                result_line, mSSIM, metric_result, seg_lines, mIoU, mean_pixel_acc = self.single_process_evalutation()
            else:
                result_line, mSSIM, metric_result, seg_lines, mIoU, mean_pixel_acc = self.single_process_evalutation()#self.multi_process_evaluation()

            results.write('Model: ' + model + '\n')
            results.write(result_line)
            results.write('\n')
            for line in seg_lines:
                results.write(line + '\n') 
            # results.write('\n')
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
            # 2024-06-10 20:01 修改输出指标
            
            # ir = cv2.split(item['modal_x'])[0]
            # print(item['data'])
            # print(item['modal_x'])
            # vi = (item['data'] * 255).astype(np.uint8)  # 一通道rgb的
            # ir = (item['modal_x'] * 255).astype(np.uint8)
            ir = (item['modal_x'] * 255).transpose(2, 0, 1)[0, :, :].astype(np.uint8)
            vi = cv2.cvtColor((item['data'] * 255).transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)

            results_dict = self.func_per_iteration(item, self.devices[0], self.config)
            fi = results_dict['output_fused']
            fi = np.transpose(fi, (1, 2, 0))
            fi = cv2.cvtColor(fi, cv2.COLOR_RGB2GRAY)

            hist += results_dict['hist']
            correct += results_dict['correct']
            labeled += results_dict['labeled']
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

            fusion_metric_result += metrics
            all_results.append(results_dict)

        fusion_metric_result /= len(all_results)
        
        result_line, mSSIM = self.compute_metric(fusion_metric_result, self.config)
        seg_lines, mIoU, mean_pixel_acc = self.compute_score(hist, correct, labeled)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mSSIM, fusion_metric_result, seg_lines, mIoU, mean_pixel_acc

    def multi_process_evaluation(self):
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))

        # start multi-process on multi-gpu
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            logger.info('GPU %s handle %d data.' % (device, len(shred_list)))

            p = self.context.Process(target=self.worker, args=(shred_list, device))
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results, self.config)

        for p in procs:
            p.join()

        result_line, mean_metric = self.compute_metric(all_results, self.config)
        logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, mean_metric
    
    def func_per_iteration(self, item, device, config):
        img_rgb = item['data']
        img_ir = item['modal_x']
        text_rgb = item['des_rgb']
        text_ir = item['des_x']
        label = item['label']
        name = item['fn']

        # pred = self.sliding_eval_rgbX(img_rgb.transpose(1, 2, 0), img_ir, self.eval_crop_size, self.eval_stride_rate, device)
        # 这点重建的时候注释掉了 2024-9-29 23:44修改
        # img_ir = np.expand_dims(img_ir, axis=2)
        # img_rgb = np.expand_dims(img_rgb, axis=2)
        
        # print(img_rgb.shape)
        # print(img_ir.shape)
        # .permute(0, 3, 1, 2)  .permute(0, 3, 1, 2)#
        # print(img_rgb.shape)
        # print(img_ir.shape)
        # img_rgb = torch.from_numpy(img_rgb).to(device).float().unsqueeze(0)
        # img_ir = torch.from_numpy(img_ir).to(device).float().unsqueeze(0)
        # print(img_rgb.shape)
        img_rgb = torch.from_numpy(img_rgb).to(device).float().unsqueeze(0)#.permute(0, 3, 1, 2)#.permute(0, 1, 2).unsqueeze(0)
        # img_ir = torch.from_numpy(img_ir).to(device).float().unsqueeze(0).permute(0, 3, 1, 2)#.permute(0, 1, 2).unsqueeze(0)
        img_ir = torch.from_numpy(img_ir).to(device).float().unsqueeze(0).permute(0, 3, 1, 2)#.permute(0, 1, 2).unsqueeze(0)
        text_rgb = torch.from_numpy(text_rgb).to(device).float().unsqueeze(0)
        text_ir = torch.from_numpy(text_ir).to(device).float().unsqueeze(0)

        with torch.no_grad():
            output_fused, output_semantic, _ = self.val_func(img_rgb, img_ir, text_rgb, text_ir)
            output_fused = (output_fused - torch.min(output_fused)) / (torch.max(output_fused) - torch.min(output_fused))
            output_semantic = torch.exp(output_semantic[0]).permute(1, 2, 0)
            # rec_rgb, rec_ir = self.val_func(img_rgb, img_ir)

        pred = output_semantic.squeeze().cpu().numpy().argmax(2)
        hist_seg, labeled_seg, correct_seg = self.hist_info(config.num_classes, pred, label)
        
        results_dict = {
            'output_fused': np.squeeze(output_fused * 255).cpu().numpy().astype(np.uint8),
            'hist': hist_seg, 
            'labeled': labeled_seg, 
            'correct': correct_seg,
        }
        
        # results_dict = {
        #     'rec_rgb': rec_rgb.squeeze().cpu().numpy(),
        #     'rec_ir': rec_ir.squeeze().cpu().numpy()
        # }


        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn_fus = name + '_fus.png'
            fn_seg = name + '_seg.png'
            fn_lab = name + '_lab.png'
            fn_com = name + '_com.png'
            # output_fused = (output_fused - torch.min(output_fused)) / (torch.max(output_fused) - torch.min(output_fused))
            
            # save fusion result
            output_fused = output_fused * 255
            fusion_img = output_fused.squeeze().cpu().numpy().astype(np.uint8)
            fusion_img = fusion_img.transpose(1, 2, 0)
            # print(fusion_img.shape)
            # fusion_img = np.transpose(fusion_img, (1, 2, 0))
            # Image.fromarray(fusion_img).save(os.path.join(self.save_path, fn))
            cv2.imwrite(os.path.join(self.save_path, fn_fus), fusion_img)
            # class_colors, color_map = self.get_class_colors()
            class_colors = config.pattale
            segment_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for i in range(config.num_classes):
                segment_img[pred == i] = class_colors[i]
                label_img[label == i] = class_colors[i]
            # for cls_id, cls_color in color_map.items():
            #     label_img[label == cls_id] = cls_color
        
            # fusion_img = cv2.cvtColor(fusion_img, cv2.COLOR_GRAY2BGR)
            
            combined_img = cv2.hconcat([fusion_img, segment_img, label_img])

            # cv2.imwrite(os.path.join(self.save_path, fn_seg), segment_img)
            # cv2.imwrite(os.path.join(self.save_path, fn_lab), label_img)
            cv2.imwrite(os.path.join(self.save_path, fn_com), combined_img)

            
            # logger.info('Save the image ' + fn)
        
        # recontruction
        # if self.save_path is not None:
        #     ensure_dir(self.save_path)
        #     fn1 = name + '_rgb.png'
        #     fn2 = name + '_ir.png'
        #     rec_rgb = (rec_rgb - torch.min(rec_rgb)) / (torch.max(rec_rgb) - torch.min(rec_rgb))
        #     rec_ir = (rec_ir - torch.min(rec_ir)) / (torch.max(rec_ir) - torch.min(rec_ir))

        #     rec_rgb = rec_rgb * 255
        #     rec_ir = rec_ir * 255

        #     rec_rgb_img = rec_rgb.squeeze().cpu().numpy().astype(np.uint8)
        #     rec_ir_img = rec_ir.squeeze().cpu().numpy().astype(np.uint8)

        #     rec_rgb_img = np.transpose(rec_rgb_img, (1, 2, 0))
        #     cv2.imwrite(os.path.join(self.save_path, fn1), rec_rgb_img)
        #     cv2.imwrite(os.path.join(self.save_path, fn2), rec_ir_img)

        return results_dict

    def get_class_colors(self):
        pattale = [
            [0, 0, 0],  # unlabelled
            [128, 0, 64],  # car
            [0, 64, 64],  # person
            [192, 128, 0],  # bike
            [192, 0, 0],  # curve
            [0, 128, 128],  # car_stop
            [128, 64, 64],  # guardrail
            [128, 128, 192],  # color_cone
            [0, 64, 192],  # bump
        ]
        color_map = {
            0: (0, 0, 0),         
            1: (128, 0, 64),       
            2: (0, 64, 64),       
            3: (192, 128, 0),     
            4: (192, 0, 0),       
            5: (0, 128, 128),     
            6: (128, 64, 64),     
            7: (128, 128, 192),   
            8: (0, 64, 192),        
        }
        return pattale, color_map
    
    def hist_info(self, n_cl, pred, gt):
        assert (pred.shape == gt.shape)
        k = (gt >= 0) & (gt < n_cl)
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))
        confusionMatrix = np.bincount(n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2).reshape(n_cl, n_cl)
        return confusionMatrix, labeled, correct

    def compute_metric(self, all_results, config):
        EN = np.round(all_results[0], 2)
        SD = np.round(all_results[1], 2)
        SF = np.round(all_results[2], 2)
        MI = np.round(all_results[3], 2)
        SCD = np.round(all_results[4], 2)
        VIFF = np.round(all_results[5], 2)
        Qabf = np.round(all_results[6], 2)
        SSIM = np.round(all_results[7], 2)

        result_line = "EN: {:.2f}, SD: {:.2f}, SF: {:.2f}, MI: {:.2f}, SCD: {:.2f}, VIFF: {:.2f}, Qabf: {:.2f}, SSIM: {:.2f}".format(EN, SD, SF, MI, SCD, VIFF, Qabf, SSIM)
        return result_line, SSIM
    
    def compute_score(self, hist, correct, labeled):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IoU = np.nanmean(iou)
        mean_IoU_no_background = np.nanmean(iou[1:]) # useless for NYUDv2

        freq = hist.sum(1) / hist.sum()
        freq_IoU = (iou[freq > 0] * freq[freq > 0]).sum()

        class_acc = np.diag(hist) / hist.sum(axis=1)
        mean_pixel_acc = np.nanmean(class_acc)

        pixel_acc = correct / labeled

        seg_lines = []
        seg_lines.append('%-10s\t%-10s\t%-10s' % ('Class', 'IoU (%)', 'Class Acc (%)'))
        for i in range(self.class_num):
            cls = '%d %s' % (i+1, self.config.class_names[i])
            seg_lines.append('%-10s\t%-10.3f%%\t%-10.3f%%' % (cls, iou[i] * 100, class_acc[i] * 100))
        seg_lines.append('%-10s\t%-10.3f%%\t%-10s\t%10.3f%%\t%-10s\t%10.3f%%\t%-10s\t%10.3f%%\t%-10s\t%10.3f%%' % (
            'mean_IoU:', mean_IoU * 100, 
            'mean_IU_no_back:', mean_IoU_no_background*100,
            'freq_IoU:', freq_IoU*100, 
            'mean_pixel_acc:', mean_pixel_acc*100, 
            'pixel_acc:', pixel_acc*100
            ))
        seg_lines.append('----------' + '-' * 27)
        return seg_lines, mean_IoU, mean_pixel_acc #iou, mean_IoU, mean_IoU_no_background, freq_IoU, mean_pixel_acc, pixel_acc, class_acc

    def worker(self, shred_list, device):
        start_load_time = time.time()
        logger.info('Load Model on Device %d: %.2fs' % (device, time.time() - start_load_time))

        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device, self.config)
            self.results_queue.put(results_dict)


    def sliding_eval_rgbX(self, img, modal_x, crop_size, stride_rate, device=None):
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
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

        if new_cols <= crop_size[1] or new_rows <= crop_size[0]:
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

                    input_data, input_modal_x, tmargin = self.process_image_rgbX(img_sub, modal_x_sub, crop_size)
                    temp_score = self.val_func_process_rgbX(input_data, input_modal_x, device)
                    
                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                            tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

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
    
        p_img = normalize(p_img)
        p_modal_x = normalize(p_modal_x)
    
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_modal_x, _ = pad_image_to_shape(p_modal_x, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)
            if len(modal_x.shape) == 2:
                p_modal_x = p_modal_x[np.newaxis, ...]
            else:
                p_modal_x = p_modal_x.transpose(2, 0, 1) # 3 H W
        
            return p_img, p_modal_x, margin
    
        p_img = p_img.transpose(2, 0, 1) # 3 H W

        if len(modal_x.shape) == 2:
            p_modal_x = p_modal_x[np.newaxis, ...]
        else:
            p_modal_x = p_modal_x.transpose(2, 0, 1)
    
        return p_img, p_modal_x
