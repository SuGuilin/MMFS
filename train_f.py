import os
import sys
sys.path.append("/home/suguilin/VMamba/")
import time
import argparse
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from dataloader.dataloader import get_train_loader
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre

from utils.loss import Fusionloss, cc, SSIMLoss, RGBLoss, total_variation_loss, DegradationLoss, MakeFusionLoss
from utils.lr_policy import WarmUpPolyLR
from utils.logger import get_logger
from utils.init_func import group_weight
from utils.engine import Engine
from utils.engine import all_reduce_tensor, parse_devices
from utils.evaluator import Evaluator
from utils.print_indicators import Print_Indicators

from model.MYFusion import MYFusion
from model.MMoEFusion import MMoEFusion
from encoder.degradetion import DegradationModel
import torch.onnx

import kornia
import warnings
from memory_profiler import profile

import numpy as np
from PIL import Image
import cv2

from tensorboardX import SummaryWriter
warnings.filterwarnings("ignore")

os.environ['MASTER_PORT'] = '16005'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
parser = argparse.ArgumentParser()
logger = get_logger()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    dataset_name = args.dataset_name
    if dataset_name == 'mfnet':
        config_path = './configs/config_mfnet.yaml'
    elif dataset_name == 'fmb':
        config_path = './configs/config_fmb.yaml'
    else:
        raise ValueError('Not a valid dataset name')
    
    config = engine.load_config(config_path)
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)

    if not engine.distributed:
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        logger.info(f'TensorBoard logging to {tb_dir}')
        # engine.link_tb(tb_dir, generate_tb_dir)


    ssim_fusion = SSIMLoss(window_size=11, size_average=True)
    criterion_fusion = Fusionloss()
    # criterion_fusion = MakeFusionLoss()

    L1Loss = nn.L1Loss()
    Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')
    MSELoss = nn.MSELoss()
    DegLoss = DegradationLoss()
    Seg_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    # PerceptLoss = PerceptualLoss()

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 
    # model = MYFusion()
    model = MMoEFusion(
        device=device, 
        num_classes=config.num_classes, 
        embed_dim=config.decoder_embed_dim, 
        seg_norm=BatchNorm2d,
        align_corners=config.align_corners,
    )
    degradation = DegradationModel()
    model_name = 'MMoEFusion'
    
    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    
    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        degradation.to(device)
    
    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')

    val_setting = {'rgb_root': config.rgb_root_folder,
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
    val_dataset = RGBXDataset(val_setting, 'val', val_pre, labelflag=True)

    best_mean_metric = 0.0
    best_epoch = 100000

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)
        sum_loss = 0
        sum_loss_seg = 0
        sum_loss_fus = 0
        # sum_loss_deg = 0
        sum_loss_gui = 0
        sum_loss_aux = 0
        
        for idx in pbar:
            engine.update_iteration(epoch, idx)
            minibatch = next(dataloader)
            imgs = minibatch['data']
            modal_xs = minibatch['modal_x']
            name = minibatch['fn']
            des_rgb = minibatch['des_rgb']
            des_x = minibatch['des_x']
            guide = minibatch['guide']
            label = minibatch['label']


            # imgs = imgs * 255
            # imgs = imgs.squeeze().cpu().numpy().astype(np.uint8)
            # imgs = imgs.transpose(1, 2, 0)
            # cv2.imwrite('rgb.png', imgs)

            # modal_xs = modal_xs * 255
            # modal_xs = modal_xs.squeeze().cpu().numpy().astype(np.uint8)
            # modal_xs = modal_xs.transpose(1, 2, 0)
            # cv2.imwrite('ir.png', modal_xs)
            # print(modal_xs.shape)
            # print(imgs.shape)
            # print(label.shape)

            imgs = imgs.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            des_rgb = des_rgb.cuda(non_blocking=True)
            des_x = des_x.cuda(non_blocking=True)
            guide = guide.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            # rgb = degradation(imgs)
            # ir = degradation(modal_xs, infrared=True)
            rgb = imgs
            ir = modal_xs
            fused, seg_out, loss_aux = model(rgb, ir, des_rgb, des_x) #, loss_load, deg_rgb, deg_ir
            # print(seg_out.shape)
            # print(torch.min(seg_out), torch.max(seg_out))
            # loss_fusion, loss_in, loss_grad = criterion_fusion(imgs, modal_xs, fused)
            # MRSF的损失函数
            loss_fusion = criterion_fusion(imgs, modal_xs, fused, config)
            # loss_deg = MSELoss(imgs, deg_rgb) + MSELoss(modal_xs, deg_ir) #DegLoss(param_rgb) + DegLoss(param_ir)
            # reg_fused = (fused - torch.min(fused)) / (torch.max(fused) - torch.min(fused))
            # loss_guide = L1Loss(reg_fused, guide)
            loss_seg = Seg_loss(seg_out, label)
            # 试一下将引导系数改为1， 分割改为20
            # loss = loss_fusion +  2 * loss_guide + 10 * loss_seg #+ 0.1*loss_deg#+ 2*(2 - Loss_ssim(imgs, fused) - Loss_ssim(modal_xs, fused)) #+ loss_load                             # + 0.2 * ssim_fusion(imgs, modal_xs, fused)
            #loss = loss_fusion +  2 * loss_guide + 15 * loss_seg #+ 0.1*loss_deg#+ 2*(2 - Loss_ssim(imgs, fused) - Loss_ssim(modal_xs, fused)) #+ loss_load                             # + 0.2 * ssim_fusion(imgs, modal_xs, fused)
            loss = config.theta * loss_seg + loss_fusion + config.sigma * loss_aux
            # recontruction 
            # imgs_rec, modal_xs_rec = model(imgs, modal_xs)
            # loss = MSELoss(imgs_rec, imgs) + MSELoss(modal_xs_rec, modal_xs) + L1Loss(imgs_rec, imgs) + L1Loss(modal_xs_rec, modal_xs)
            # loss += 2 - Loss_ssim(imgs, imgs_rec) - Loss_ssim(modal_xs, modal_xs_rec)
            
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                semantic_loss = all_reduce_tensor(loss_seg, world_size=engine.world_size)
                fusion_loss = all_reduce_tensor(loss_fusion, world_size=engine.world_size)
                aux_loss = all_reduce_tensor(loss_aux, world_size=engine.world_size)
                # degraded_loss = all_reduce_tensor(loss_deg, world_size=engine.world_size)
                #guide_loss = all_reduce_tensor(loss_guide, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                if dist.get_rank() == 0:
                    sum_loss += reduce_loss.item()
                    sum_loss_seg += semantic_loss.item()
                    sum_loss_fus += fusion_loss.item()
                    sum_loss_aux += aux_loss.item()
                    # sum_loss_deg += degraded_loss.item()
                    # sum_loss_gui += guide_loss.item()

                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (semantic_loss.item(), (sum_loss_seg / (idx + 1))) \
                            + ' loss_fus=%.4f total_loss_fus=%.4f' % (fusion_loss.item(), (sum_loss_fus / (idx + 1))) \
                            + ' loss_aux=%.4f total_loss_aux=%.4f' % (aux_loss.item(), (sum_loss_gui / (idx + 1))) \
                            # + ' loss_deg=%.4f total_loss_deg=%.4f' % (degraded_loss.item(), (sum_loss_deg / (idx + 1))) \
                    pbar.set_description(print_str, refresh=False)
            else:
                sum_loss += loss
                sum_loss_seg += loss_seg
                sum_loss_fus += loss_fusion
                sum_loss_aux += loss_aux
                # sum_loss_deg += loss_deg
                #sum_loss_gui += loss_guide
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                        + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.4e' % lr \
                        + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1))) \
                        + ' loss_seg=%.4f' % (loss_seg) \
                        + ' loss_fus=%.4f' % (loss_fusion) \
                        + ' loss_aux=%.4f' % (loss_aux) \
                        # + ' loss_deg=%.4f total_loss_deg=%.4f' % (loss_deg, (sum_loss_deg / (idx + 1))) \
                pbar.set_description(print_str, refresh=False)
            del loss, loss_seg, loss_fusion, loss_aux#, loss_guide #,loss_deg

        if not engine.distributed:
            tb.add_scalar('train/total_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('train/fusion_loss', sum_loss_fus / len(pbar), epoch)
            tb.add_scalar('train/segment_loss', sum_loss_seg / len(pbar), epoch)
            tb.add_scalar('train/aux_loss', sum_loss_aux / len(pbar), epoch)
            tb.flush()
            
        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if not engine.distributed or (engine.distributed and engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)
        
        # devices_val = [engine.local_rank] if engine.distributed else [0]
        torch.cuda.empty_cache()

        if engine.distributed:
            if dist.get_rank() == 0:
                # only test on rank 0, otherwise there would be some synchronization problems
                # evaluation to decide whether to save the model
                if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                    model.eval() 
                    with torch.no_grad():
                        all_dev = parse_devices(args.devices)
                        fuser = Evaluator(dataset=val_dataset, save_path=config.save_path, #'./results_moe',
                                                network=model, devices=[model.device],
                                                verbose=False, config=config,)
                        metric_result, mean_metric, mSSIM, mIoU, mean_pixel_acc = fuser.run(config.checkpoint_dir, str(epoch), config.val_log_file)#, config.link_val_log_file)
                        print('mean_SSIM:', mSSIM, 'mean_IoU:', mIoU, 'mean_pixel_acc:', mean_pixel_acc)
                        
                        # Determine if the model performance improved
                        if mIoU > best_mean_metric:
                            # If the model improves, remove the saved checkpoint for this epoch
                            last_best_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                            # if os.path.exists(last_best_path):
                            #     os.remove(last_best_path)
                            engine.save_checkpoint(checkpoint_path)
                            best_epoch = epoch
                            best_mean_metric = mIoU
                            Print_Indicators(model_name, metric_result)
                        else:
                            # If the model does not improve, remove the saved checkpoint for this epoch
                            checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                            if os.path.exists(checkpoint_path):
                                os.remove(checkpoint_path)
                    model.train()
        else:
            if (epoch >= config.checkpoint_start_epoch) and (epoch - config.checkpoint_start_epoch) % config.checkpoint_step == 0:
                model.eval() 
                with torch.no_grad():
                    devices_val = [engine.local_rank] if engine.distributed else [0]
                    fuser = Evaluator(dataset=val_dataset, save_path=config.save_path, #'./experiment/exp2/result',
                                            network=model, devices=[0, 1, 2, 3],
                                            verbose=False, config=config,
                                            )
                    metric_result, mean_metric, mSSIM, mIoU, mean_pixel_acc = fuser.run(config.checkpoint_dir, str(epoch), config.val_log_file)#, config.link_val_log_file)
                    print('mean_SSIM:', mSSIM, 'mean_IoU:', mIoU, 'mean_pixel_acc:', mean_pixel_acc)
                    
                    # Determine if the model performance improved
                    if mIoU > best_mean_metric:
                        # If the model improves, remove the saved checkpoint for this epoch
                        last_best_path = os.path.join(config.checkpoint_dir, f'epoch-{best_epoch}.pth')
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'best_model.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                        # if os.path.exists(last_best_path):
                        #     os.remove(last_best_path)
                        engine.save_checkpoint(checkpoint_path)
                        best_epoch = epoch
                        best_mean_metric = mIoU
                        Print_Indicators(model_name, metric_result)
                    else:
                        # If the model does not improve, remove the saved checkpoint for this epoch
                        checkpoint_path = os.path.join(config.checkpoint_dir, f'epoch-{epoch}.pth')
                        if os.path.exists(checkpoint_path):
                            os.remove(checkpoint_path)
                model.train()