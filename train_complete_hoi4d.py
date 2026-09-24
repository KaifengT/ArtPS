import torch.optim as optim
import torch
from backbone.PointComplete.utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
import time, datetime
from tools.utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from backbone.PointComplete.models.PointAttN_PP import Model

from dataloader.hoi4d_dataset.hoi4d_dataset_arti import HOI4D, get_category_name, get_num_parts, get_category_jointtype



def train():
    logging.info(str(args))
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}



    noise_aug_scale = 0.002
    shape_aug_scale = 0.1
    base_aug_maxd = 60.
    base_aug_maxt = 0.4
    
    

    train_dataset = HOI4D(root_dir=DATA_ROOT,
                                category=CATEGOIES,
                                num_points=NUM_POINTS,
                                mode='train', 
                                sdf_mode=False,
                                add_noise=False,
                                use_cache=True,
                                force_write_cache=False,
                                cache_dir=DATA_CACHE
                            )

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers),
                                                    drop_last=True)

    test_dataset = HOI4D(root_dir=DATA_ROOT,
                                category=CATEGOIES,
                                num_points=NUM_POINTS,
                                mode='test', 
                                sdf_mode=False,
                                add_noise=False,
                                use_cache=True,
                                force_write_cache=False,
                                cache_dir=DATA_CACHE
                            )

    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers),
                                                    )
    
    
    
    logging.info('Length of train dataset:%d', len(train_dataset))
    logging.info('Length of test dataset:%d', len(test_dataset))
    
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    seed = 589
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(Model(num_parts=NUM_PARTS, r2=NUM_POINTS//1024))
    net.cuda()
    # if hasattr(model_module, 'weights_init'):
    #     net.module.apply(model_module.weights_init)
    
    lr = args.lr

    
    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, betas=betas)


    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(dataloader), epochs=args.nepoch+1, pct_start=0.3, max_lr=lr, final_div_factor=0.4)


    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)
    
    for epoch in range(args.start_epoch, args.nepoch):
    
        train_loss_meter.reset()
        net.module.train()

    
        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
    
            data = go_to_device(data, 'cuda')
            cloud, gt_part_cls, gt_rt, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, limits, gt_complete_cloud, complete_cloud_seg, gt_complete_cloud_per_part, video_name, frame_id, _ = data
    
            ''' DATA AUGMENTATION ''' 
            '''
            gt_complete_cloud gt_complete_cloud_per_part not augmented
            '''
            B, P, _, _ = gt_rt.size()
            noise = torch.randn_like(cloud) * 0.0001
            cloud += noise
                        
            '''
            SIZE ARGUMENTATION
            FIXME WARNING if part not connect to the base, the augmentation will be wrong
            '''
            
            scale_ratio = None
            for part_id in range(P):
                
                _norm_joint_loc = gt_norm_joint_loc[:, part_id-1] if part_id > 0 else None
                _parent_rt = gt_rt[:, 0] if part_id > 0 else None
                
                scaled_part_rt, \
                scaled_part_norm_joint_loc, \
                scale_ratio = size_argumentation_2(scale_ratio=scale_ratio,
                                     parent_rt=_parent_rt,
                                     part_rt=gt_rt[:, part_id],
                                     part_norm_joint_loc=_norm_joint_loc,
                                     scale_range=0.1,
                                     return_scale_ratio=True,
                                     )
                
                gt_rt[:, part_id] = scaled_part_rt
                
                
                # prismatic joints need scale joint state
                if part_id > 0 and JOINTS_TYPE[part_id-1] == 'prismatic':
                    state_scale = torch.linalg.norm(gt_norm_joint_axis[:, part_id-1:part_id] * scale_ratio[:, None, :], dim=-1)
                    gt_joint_state[:, part_id] *= state_scale.squeeze(-1)
                
                
                if part_id > 0:
                    gt_norm_joint_loc[:, part_id-1] = scaled_part_norm_joint_loc
                
            # scale cloud

            scale_transform = torch.eye(4, device=gt_rt.device).unsqueeze(0).expand_as(scaled_part_rt).clone()
            scale_transform[..., :3, :3] = torch.diag_embed(scale_ratio, dim1=-2, dim2=-1)
            scale_transform_wpose = gt_rt[:, 0] @ scale_transform @ torch.linalg.inv(gt_rt[:, 0])
            
            cloud = transform_points(scale_transform_wpose, cloud)
            gt_complete_cloud *= scale_ratio.unsqueeze(-2)
            '''
            BASE POSE ARGUMENTATION
            FIXME WARNING if part not connect to the base, the augmentation will be wrong
            '''
            for part_id in range(P):    
                if part_id == 0:
                    ...
                    cloud, gt_rt = base_augmentation(cloud=cloud, gt_part_rt=gt_rt, maxd=60, maxt=0.01)
                
                elif part_id > 0:
                    ...
                    # cloud, gt_rt, gt_joint_state = part_augmentation(cloud, gt_rt, part_id, gt_part_cls, gt_norm_joint_loc, gt_norm_joint_axis, gt_joint_state,
                    #                                                 #  min_angle=limits[:, part_id, 0:1], max_angle=limits[:, part_id, 1:2],
                    #                                                 min_angle=0, max_angle=0,
                    #                                                  joint_type=self.JTYPE[part_id-1],
                    #                                                  )
                    
                    
                    
            gt_observed_joint_axis = transform_points_33(gt_rt[:, 0, :3, :3], gt_norm_joint_axis)
            gt_observed_joint_loc = transform_points(gt_rt[:, 0], gt_norm_joint_loc)
                    
            gt_complete_cloud_per_part = transform_points(gt_rt @ scale_transform[:, None], gt_complete_cloud_per_part)

            
            # visualdata = {
            #     'inputs': cloud.detach().cpu().numpy(),
            #     # 'gt': gt_centroid.detach().cpu().numpy(),
            #     'video_name': video_name,
            #     'frame_id': frame_id,
            # }
            # for part_id in range(P):
            #     visualdata.update({
            #         f'gt_part_{part_id}': gt_complete_cloud_per_part[:, part_id, ...].detach().cpu().numpy(),
            #     })
            # with open(f'tempfile/{i}.pkl', 'wb') as f:
            #     pickle.dump(visualdata, f)
            # if i > 20:exit()
            # continue

    
            # inputs = inputs.float().cuda()
            # gt = gt.float().cuda()

            out2, loss2, net_loss = net(x=cloud,
                                        x_cls=gt_part_cls,
                                        gt=gt_complete_cloud_per_part,
                                        )
    
            train_loss_meter.update(net_loss.mean().item())
    
            net_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            
            torch.nn.utils.clip_grad_norm_(net.module.parameters(), 1.0)
    
            optimizer.step()
            scheduler.step()
    
            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f, total_loss: %f lr: %f' %
                             (epoch, i, len(train_dataset) / args.batch_size, args.loss, loss2.mean().item(), net_loss.mean().item(), optimizer.param_groups[0]['lr']))
    
        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")
    
        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses)
            # plot_cd_vs_epoch(epoch_cd)

def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()
    
    color = torch.tensor([
    [.78, .32, .16],
    [.90, .76, .52],
    [.98, .95, .77],
    [.45, .66, .57],
    [0.0, .52, .52],
    [0.0, 0.0, 0.0],
                ], device='cuda')

    collector = Collector(auto_convert=True)

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            data = go_to_device(data, 'cuda')
            cloud, gt_part_cls, gt_rt, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, limits, gt_complete_cloud, complete_cloud_seg, gt_complete_cloud_per_part, video_name, frame_id = data


            gt_complete_cloud_per_part = transform_points(gt_rt, gt_complete_cloud_per_part)


    
            result_dict = net(cloud, gt_part_cls, gt_complete_cloud_per_part, is_training=False)
            
            for k, v in val_loss_meters.items():
                v.update(result_dict[k].mean().item())
                
            collector.add('gt', gt_complete_cloud_per_part)
            collector.add('gt_cls', gt_part_cls)
            collector.add('inputs', cloud)
            collector.add('pred', result_dict['out2'])
            
        collector.compose()

        epoch_cd['cd_p'].append(val_loss_meters['cd_p'].avg)
        epoch_cd['cd_t'].append(val_loss_meters['cd_t'].avg)
        epoch_cd['cd_t_coarse'].append(val_loss_meters['cd_t_coarse'].avg)
        epoch_cd['cd_p_coarse'].append(val_loss_meters['cd_p_coarse'].avg)
    
        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                
                # collector.saveh5(os.path.join(log_dir, f'eval_result_best.h5'))
                collector.save(os.path.join(log_dir, f'eval_result_best.pkl'))
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)
    
        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)
    
        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('--config', help='path to config file', default='configs/config_complete_net_hoi4d.yaml')
    parser.add_argument('-c', '--category', type=str, default='C4', help='Object category.', choices=['C3', 'C4', 'C6', 'C8', 'C14'])
    parser.add_argument('-n', '--num_points', type=int, default=2048, help='Number of points to sample.')
    parser.add_argument('--hoi4d_dataset_root', type=str, help='Root path of the Hoi4D dataset.')
    parser.add_argument('--hoi4d_dataset_cache', type=str, help='Cache path of the Hoi4D dataset.')


    arg = parser.parse_args()
    config_path = os.path.join(arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    
    CATEGOIES = arg.category
    NUM_POINTS = arg.num_points
    DATA_ROOT = arg.hoi4d_dataset_root
    DATA_CACHE = arg.hoi4d_dataset_cache
    NUM_PARTS = get_num_parts(CATEGOIES)
    JOINTS_TYPE = get_category_jointtype(CATEGOIES)
    CATEGOIE_NAME = get_category_name(CATEGOIES)
    
    epoch_cd = {'cd_p': [], 'cd_t': [], 'cd_t_coarse': [], 'cd_p_coarse': []}

    work_dir = os.path.join(args.work_dir, f'cmp_hoi4d_{CATEGOIE_NAME}')
    
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    
    else:
        exp_name = f'{args.model_name}_{args.loss}_{CATEGOIES}_{args.ratio1}{args.ratio2}_hoi4d_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}'
        log_dir = os.path.join(work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print('save_path:', work_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)],
                        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
                        force=True)
    train()



