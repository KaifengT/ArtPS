import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tools.utils import *
from tqdm import tqdm
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from backbone.mamba3d.mamba3d import Mamba3D, Mamba3d_config
import math
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from models import import_model_module
import json
# from models.sdf_model_V10 import *


class RandomCoveringSampler(Sampler):
    """
    自定义采样器，每次随机抽取n个样本，并确保最终覆盖所有数据
    """
    def __init__(self, data_source, samples_per_epoch=None):
        self.data_source = data_source
        self.total_size = len(self.data_source)
        # 如果未指定每个epoch的样本数，默认使用全部
        self.samples_per_epoch = samples_per_epoch or self.total_size
        # 已采样过的样本索引
        self.sampled_indices = set()
        
    def __iter__(self):
        # 如果所有样本都已经被采样过或者是第一次采样，重置状态
        if len(self.sampled_indices) == self.total_size:
            self.sampled_indices = set()
        
        remaining_indices = list(set(range(self.total_size)) - self.sampled_indices)
        
        # 计算本次需要采样的数量
        n_to_sample = min(self.samples_per_epoch, len(remaining_indices))
        
        # 随机采样
        indices = np.random.choice(remaining_indices, n_to_sample, replace=False).tolist()
        
        # 更新已采样索引
        self.sampled_indices.update(indices)
        
        return iter(indices)
    
    def __len__(self):
        return min(self.samples_per_epoch, self.total_size - len(self.sampled_indices))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_kl_loss(mu, logvar):
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def eikonal_loss(pred_sdf, points):
    
    # points.requires_grad_(True)
    
    gradients = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        retain_graph=True
    )[0]
    
    grad_norm = torch.norm(gradients, dim=-1)
    
    # Eikonal: |∇f(x) - 1|^2
    eik_loss = torch.mean((grad_norm - 1.0) ** 2)
    
    return eik_loss

def augment_scale(cloud:torch.tensor, 
                  sample:torch.tensor, 
                  gt_sdf:torch.tensor, 
                  gt_joint_loc:torch.tensor,
                  gt_bbox_transforms:torch.tensor=None,
                  gt_bbox_sizes:torch.tensor=None,
                  aug_range:float=0.2):
    
    scale_factors = torch.empty((cloud.shape[0], 1, 1), device=cloud.device).uniform_(1.0 - aug_range, 1.0 + aug_range) # B, 1, 1
    
    cloud = cloud * scale_factors # B, N, 3
    sample = sample * scale_factors # B, N, 3
    gt_joint_loc = gt_joint_loc * scale_factors
    gt_sdf = gt_sdf * scale_factors.squeeze(-1)
    #NOTE  gt_joint_axis does not change
    
    
    if gt_bbox_transforms is not None and gt_bbox_sizes is not None:
        gt_bbox_transforms[..., :3, 3] = gt_bbox_transforms[..., :3, 3] * scale_factors
        gt_bbox_sizes = gt_bbox_sizes * scale_factors
        
    return cloud, sample, gt_sdf, gt_joint_loc, gt_bbox_transforms, gt_bbox_sizes
    
    
    
def augment_pose(cloud:torch.tensor,
                 sample:torch.tensor,
                 gt_joint_loc:torch.tensor,
                 gt_joint_axis:torch.tensor,
                 gt_bbox_transforms:torch.tensor=None,
                 maxd:float=5.0, 
                 maxt:float=0.15):
    
    
    def gen_noise_uniform_transform(B, device, maxd, maxt):
        
        noise_transform = torch.eye(4).to(device)[None, :, :].repeat(B, 1, 1)
        
        if maxd > 0:
            angle = torch.empty((B, 1), device=device).uniform_(-maxd / 180. * np.pi, maxd / 180. * np.pi)
            axis = torch.empty((B, 3), device=device).uniform_(-1.0, 1.0)
            noise_rot = axis_angle_to_matrix(axis * angle)
            noise_transform[:, :3, :3] = noise_rot

        if maxt > 0:
            noise_trans = torch.empty((B, 3), device=device).uniform_(-maxt, maxt)
            noise_transform[:, :3, 3] = noise_trans

        return noise_transform
    
    
    
    B = cloud.shape[0]
    noise_pose = gen_noise_uniform_transform(B, cloud.device, maxd=maxd, maxt=maxt) # B 4 4
    
    cloud = transform_points(noise_pose, cloud)
    sample = transform_points(noise_pose, sample)
    gt_joint_loc = transform_points(noise_pose, gt_joint_loc)
    gt_joint_axis = transform_points_33(noise_pose[..., :3, :3], gt_joint_axis)
    
                
    if gt_bbox_transforms is not None:
        gt_bbox_transforms = noise_pose.unsqueeze(1) @ gt_bbox_transforms

    return cloud, sample, gt_joint_loc, gt_joint_axis, gt_bbox_transforms


def random_sample_within_unit_cube_P(sample, sdf, num_samples=4096):
    """
    Args:
        sample: (B, P, N, 3)
        sdf:    (B, P, N, 1)
        num_samples: int, default 4096

    Returns:
        sampled_sample: (B, P, num_samples, 3)
        sampled_sdf:    (B, P, num_samples, 1)
    """
    B, P, N, _ = sample.shape
    device = sample.device

    # Step 1: Create mask for points within [-1, 1] in all xyz dimensions
    # Shape: (B, P, N)
    mask = (sample >= -1.0) & (sample <= 1.0)
    mask = mask.all(dim=-1)  # (B, P, N)

    sampled_samples = []
    sampled_sdfs = []

    for b in range(B):
        batch_sample_list = []
        batch_sdf_list = []
        for p in range(P):
            # Get valid indices for this (b, p)
            valid_indices = torch.nonzero(mask[b, p], as_tuple=True)[0]  # (M,)
            M = valid_indices.numel()

            if M == 0:
                # No valid points; fallback: sample from all points (or raise error)
                # Here we just take first num_samples (should not happen if data is well-formed)
                selected_indices = torch.arange(num_samples, device=device) % N
            elif M < num_samples:
                # Not enough valid points: sample with replacement
                selected_indices = valid_indices[torch.randint(M, (num_samples,), device=device)]
            else:
                # Enough valid points: sample without replacement
                perm = torch.randperm(M, device=device)
                selected_indices = valid_indices[perm[:num_samples]]

            # Gather sampled points
            sampled_sample = sample[b, p, selected_indices]  # (num_samples, 3)
            sampled_sdf = sdf[b, p, selected_indices]        # (num_samples, 1)

            batch_sample_list.append(sampled_sample)
            batch_sdf_list.append(sampled_sdf)

        # Stack along P dimension
        batch_samples = torch.stack(batch_sample_list, dim=0)  # (P, num_samples, 3)
        batch_sdfs = torch.stack(batch_sdf_list, dim=0)        # (P, num_samples, 1)

        sampled_samples.append(batch_samples)
        sampled_sdfs.append(batch_sdfs)

    # Stack along B dimension
    final_sample = torch.stack(sampled_samples, dim=0)  # (B, P, num_samples, 3)
    final_sdf = torch.stack(sampled_sdfs, dim=0)        # (B, P, num_samples, 1)

    return final_sample, final_sdf



def random_sample_within_unit_cube(sample, sdf, num_samples=4096):
    """
    Args:
        sample: (B, N, 3)
        sdf:    (B, N, 1)
        num_samples: int, default 4096

    Returns:
        sampled_sample: (B, num_samples, 3)
        sampled_sdf:    (B, num_samples, 1)
    """
    B, N, _ = sample.shape
    device = sample.device

    # Step 1: Create mask for points within [-1, 1] in all xyz dimensions
    # Shape: (B,N)
    mask = (sample >= -1.0) & (sample <= 1.0)
    mask = mask.all(dim=-1)  # (B, N)

    batch_sample_list = []
    batch_sdf_list = []

    for b in range(B):
        
        valid_indices = torch.nonzero(mask[b], as_tuple=True)[0]  # (M,)
        M = valid_indices.numel()

        if M == 0:
            # No valid points; fallback: sample from all points (or raise error)
            # Here we just take first num_samples (should not happen if data is well-formed)
            selected_indices = torch.arange(num_samples, device=device) % N
        elif M < num_samples:
            # Not enough valid points: sample with replacement
            selected_indices = valid_indices[torch.randint(M, (num_samples,), device=device)]
        else:
            # Enough valid points: sample without replacement
            perm = torch.randperm(M, device=device)
            selected_indices = valid_indices[perm[:num_samples]]

        # Gather sampled points
        sampled_sample = sample[b, selected_indices]  # (num_samples, 3)
        sampled_sdf = sdf[b, selected_indices]        # (num_samples, 1)

        batch_sample_list.append(sampled_sample)
        batch_sdf_list.append(sampled_sdf)

    # Stack along P dimension
    batch_samples = torch.stack(batch_sample_list, dim=0)  # (P, num_samples, 3)
    batch_sdfs = torch.stack(batch_sdf_list, dim=0)        # (P, num_samples, 1)


    return batch_samples, batch_sdfs

class KLScheduler:

    def __init__(self, max_value, warmup_epochs, start_epoch=0, steps_per_epoch=0):
        self.max_value = max_value
        self.warmup_epochs = warmup_epochs
        self.current_epoch = start_epoch
        self.current_steps = start_epoch * steps_per_epoch if steps_per_epoch > 0 else 0
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch if steps_per_epoch > 0 else 0
    def step(self):
        self.current_steps += 1

    def get_weight(self):
        if self.warmup_epochs <= 0:
            return self.max_value
        
        if self.current_steps >= self.warmup_steps:
            return self.max_value

        return self.max_value * (self.current_steps / self.warmup_steps)


accelerator = Accelerator()

conf = OmegaConf.load('configs/sdf/config_sdf_base.yaml')
cli_conf = OmegaConf.from_cli()
if 'config' in cli_conf.keys():
    spec_config_path = cli_conf.config
    del cli_conf['config']
    spec_config = OmegaConf.load(spec_config_path)
    cli_conf = OmegaConf.merge(spec_config, cli_conf)
args = OmegaConf.merge(conf, cli_conf)


if args.dataset == 'hoi4d':
    from dataloader.hoi4d_dataset.hoi4d_dataset_arti import HOI4D, get_category_name, get_num_parts, get_category_jointtype
    dataset_name = HOI4D.get_dataset_name()
    assert args.category in ['C3', 'C4', 'C6', 'C8', 'C14'], f'Invalid category {args.category} for HOI4D dataset.'
    
elif args.dataset == 'sapien':
    from dataloader.sapien_dataset.sapien_dataset import SapienDataset, get_category_name, get_num_parts, get_category_jointtype
    assert args.category in ['C1', 'C2', 'C3', 'C4', 'C5'], f'Invalid category {args.category} for SAPIEN dataset.'
    dataset_name = SapienDataset.get_dataset_name()
else:
    raise ValueError(f'Unknown dataset {args.dataset}. Supported datasets are: hoi4d, sapien.')


model_package = import_model_module(args.model.package)
SDFEncoder = getattr(model_package, 'SDFEncoder')
SDFDecoder = getattr(model_package, 'SDFDecoder')
MODEL_FILE = model_package.__file__
VERSION = model_package.VERSION


args.dataset_root = os.path.join(args.dataset_root, dataset_name.upper() + '_SDF')
category = args.category

category_name = get_category_name(category)
num_parts = get_num_parts(category)

num_points = args.num_points
use_eikonal_loss = not args.no_eikonal_loss
scale_aug_range = args.scale_aug_range
max_dis = args.max_dis
max_epoch = args.max_epoch
lr = args.lr

catcodeRegularizationLambda = args.catcode_regularization_lambda
inscodeRegularizationLambda = args.inscode_regularization_lambda
catcode_size = args.catcode_size
CodeInitStdDev = 1.0
eikonal_lambda = args.eikonal_lambda
inscode_size = args.inscode_size
num_catcodes = args.num_catcodes

use_pose_aug = args.use_pose_aug
predict_bbox = not args.no_bbox

joint_from_inscode = args.joint_from_inscode

accelerator.wait_for_everyone()

accelerator.print('Parameters:')
accelerator.print(readable_dict(OmegaConf.to_container(args, resolve=True)))


if accelerator.is_main_process and input('Continue? (y/n): ') not in ('y', 'Y', ''): 
    raise KeyboardInterrupt()
    
accelerator.wait_for_everyone()

root_dir = 'model_dict'
project_name = f'sdf_{dataset_name}_{category_name}'
IS_DEBUG = True if sys.gettrace() else False

if accelerator.is_main_process:
    if not IS_DEBUG and input('Enable Comet ML Experiment? (y/n): ') == 'y':
        exp, exp_name = get_comet_exp(project_name, prefix=f'CC{catcode_size}_IC{inscode_size}_LR{lr}_BS{args.batch_size}')
        exp_folder_path = project_name + '_' + exp_name
        WORK_DIR = os.path.join(root_dir, project_name, exp_folder_path)
        if len(args.message): exp.log_text(args.message)
        exp.log_code(MODEL_FILE)
        exp.log_parameters(OmegaConf.to_container(args, resolve=True))

    else:
        exp = None
        WORK_DIR = os.path.join(root_dir, project_name, 'debug')
else:
    exp = None
    WORK_DIR = None

accelerate_set_seed(args.seed, device_specific=True)

aug = 0.2
_datasets_keys = ['cloud', 'sample', 'sdf', 'urdf_id', 'joint_loc', 'joint_axis']
if predict_bbox:
    _datasets_keys += ['bbox_transforms', 'bbox_extents']
train_dataset = H5Dataset(os.path.join(args.dataset_root, category_name, f'train_sdf_{dataset_name}_{category_name}_p16384_s0.03_a{aug}_l300000_u1_onepart.h5'))
train_dataset.setKeys(*_datasets_keys)

test_dataset  = H5Dataset(os.path.join(args.dataset_root, category_name, f'val_sdf_{dataset_name}_{category_name}_p16384_s0.03_a{aug}_l6000_u1_onepart.h5'))
test_dataset.setKeys(*_datasets_keys)

train_dataloader_sampler = RandomCoveringSampler(train_dataset, samples_per_epoch=4096*accelerator.num_processes)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, persistent_workers=True, sampler=train_dataloader_sampler, drop_last=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)

encoder_params = {
    'inscode_size': inscode_size * 2,
}
decoder_params = {
    'num_catcodes': num_catcodes,
    'catcode_size': catcode_size,
    'inscode_size': inscode_size,
    'num_parts': num_parts,
    'joint_from_inscode': joint_from_inscode,
}

    
encoder = SDFEncoder(**encoder_params)
decoder = SDFDecoder(**decoder_params)



optimizer = torch.optim.Adam([
                              {'params': decoder.parameters(), 'lr': lr},
                              {'params': encoder.parameters(), 'lr': lr},
                              ])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_dataloader), epochs=max_epoch+1, pct_start=0.3, max_lr=lr, final_div_factor=0.4)
lossFunc = torch.nn.L1Loss(reduction='none')
# lossFunc = torch.nn.SmoothL1Loss(reduction='sum')


train_dataloader, test_dataloader, encoder, decoder, optimizer, scheduler = accelerator.prepare(
    train_dataloader, test_dataloader, encoder, decoder, optimizer, scheduler
)



best_loss = 1e10
best_epoch = -1

kl_scheduler = KLScheduler(max_value=inscodeRegularizationLambda, warmup_epochs=max_epoch//3, start_epoch=0, steps_per_epoch=len(train_dataloader))


for epoch in range(0, max_epoch):
    
    encoder.train()
    decoder.train()
    train_count = 0

    bar = tqdm(train_dataloader, desc='Training', ascii=True, leave=False, disable=not accelerator.is_main_process) 
    
    collector = Collector(auto_convert=True)

    for i, data in enumerate(train_dataloader):
        
        with accelerator.accumulate(encoder, decoder):
        
            ''' 1. Prepare data '''
            if predict_bbox:
                cloud, sample, gt_sdf, uid, gt_joint_loc, gt_joint_axis, gt_bbox_transforms, gt_bbox_sizes = data
                gt_bbox_transforms = gt_bbox_transforms.to(torch.float32)
                gt_bbox_sizes = gt_bbox_sizes.to(torch.float32)
            else:
                cloud, sample, gt_sdf, uid, gt_joint_loc, gt_joint_axis = data
                gt_bbox_transforms = None
                gt_bbox_sizes = None
                    
            cloud = cloud.to(torch.float32)
            sample = sample.to(torch.float32)
            gt_sdf = gt_sdf.to(torch.float32)
            gt_joint_loc = gt_joint_loc.to(torch.float32)
            gt_joint_axis = gt_joint_axis.to(torch.float32)


            
            with torch.no_grad():
                
                
                # sample, gt_sdf = shuffle_sample(sample, gt_sdf)
                
                # # for ONE PART
                
                cloud_noise = torch.randn_like(cloud) * 0.01
                cloud += cloud_noise
                
                gt_sdf = gt_sdf[:, 0]
                sample = sample[:, 0]
                
                # data augmentation
                cloud, sample, gt_sdf, gt_joint_loc, gt_bbox_transforms, gt_bbox_sizes = augment_scale(
                    cloud, sample, gt_sdf, gt_joint_loc, 
                    gt_bbox_transforms, gt_bbox_sizes,
                    aug_range=scale_aug_range
                )
                
                if use_pose_aug:
                    cloud, sample, gt_joint_loc, gt_joint_axis, gt_bbox_transforms = augment_pose(
                        cloud, sample, gt_joint_loc, gt_joint_axis, 
                        gt_bbox_transforms,
                        maxd=0.0, maxt=0.05
                    )

                
                # re-indexing
                sample, gt_sdf = random_sample_within_unit_cube(sample, gt_sdf, num_samples=4096)

                
            
                                
                # color_source = normlize(gt_sdf[..., None], dim=-2)
                # rgba = torch.from_numpy(color_map_numpy(color_source.detach().cpu().numpy())).to(device=accelerator.device)
                # rgba_dir = torch.where(gt_sdf[..., None] >= 0, 
                #                       torch.tensor([0.0, 1.0, 0.0, 0.2], device=accelerator.device), 
                #                       torch.tensor([1.0, 0.0, 0.0, 1.0], device=accelerator.device))
                # collector.add('cloud', cloud)
                # collector.add('cloud_noise', cloud + cloud_noise)
                # collector.add('sample_distance', torch.cat([sample, rgba], dim=-1))
                # collector.add('sample_distance_dir', torch.cat([sample, rgba_dir], dim=-1))
                # collector.add('axis_line', torch.cat([gt_joint_loc, gt_joint_loc + gt_joint_axis], dim=1))
                # bbox_temp = torch.tensor([
                #     [[-1, -1,  1],
                #      [-1,  1,  1],
                #      [ 1,  1,  1],
                #      [ 1, -1,  1],
                     
                #      [-1, -1, -1],
                #      [-1,  1, -1],
                #      [ 1,  1, -1],
                #      [ 1, -1, -1],]
                #     ], dtype=gt_bbox_transforms.dtype, device=gt_bbox_transforms.device) / 2.
                # bbox_temp = repeat(bbox_temp, "b N C -> (B b) np N C", B=gt_bbox_transforms.shape[0], np=gt_bbox_transforms.shape[1])
                # bbox = bbox_temp * gt_bbox_sizes.unsqueeze(-2)
                # bbox = transform_points(gt_bbox_transforms, bbox)
                # collector.add('bbox', bbox)
                

            

            num_samples = sample.shape[-2]
            sample.requires_grad_(True)
            
            
            ''' 2. Forward pass '''
            
            instance_code = encoder(cloud)
            mu, logvar = instance_code.chunk(2, dim=-1)
            z = reparameterize(mu, logvar)
            
            sdf_meta = decoder(z, sample)
            sdf_pred, norm_joint_loc, norm_joint_axis = sdf_meta['sdf'], sdf_meta['norm_joint_loc'], sdf_meta['norm_joint_axis']


            ''' 3. Compute losses '''
            
            loss_sdf = lossFunc(torch.clamp(sdf_pred.squeeze(-1), max=max_dis, min=-max_dis),
                                torch.clamp(gt_sdf, max=max_dis, min=-max_dis)) / num_samples
            loss_sdf = loss_sdf.sum()
            
            loss_joint = decoder.module.joint_loss(
                norm_joint_loc, norm_joint_axis, gt_joint_loc, gt_joint_axis, get_category_jointtype(category)
            )
            
            if predict_bbox:
                norm_bbox_center, norm_bbox_size = sdf_meta['norm_bbox_center'], sdf_meta['norm_bbox_size']
                bbox_loss = decoder.module.bbox_loss(
                    norm_bbox_center, norm_bbox_size, gt_bbox_transforms[..., :3, 3], gt_bbox_sizes
                )
            else:
                bbox_loss = torch.tensor([0.], device=accelerator.device, dtype=loss_sdf.dtype)
            
            # l2_size_loss = torch.sum(torch.norm(decoder.module.catcode.weight, dim=1))
            # reg_loss = (
            #     catcodeRegularizationLambda * min(1, epoch / 20) * l2_size_loss
            # ) / num_samples
            # reg_loss = decoder.module.catcode_regularization(l2_weight=0, diversity_weight=1e-2)
            
            # Test: 0.001 -> 0.0001
            # code_loss = (instance_code * instance_code).mean() * inscodeRegularizationLambda # 0.001
            kl_weight = kl_scheduler.get_weight()
            kl_loss = compute_kl_loss(mu, logvar) * kl_weight

            loss = loss_sdf + kl_loss + loss_joint + bbox_loss
            
            
            if use_eikonal_loss:
                eik_loss = eikonal_loss(sdf_pred, sample) * eikonal_lambda
                loss += eik_loss
            else:
                # eik_loss = torch.tensor([0.], device=accelerator.device, dtype=loss.dtype)
                eik_loss = eikonal_loss(sdf_pred, sample) * eikonal_lambda
            

            ''' 4. Backpropagation '''
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(decoder.module.parameters(), 1.0)
                accelerator.clip_grad_norm_(encoder.module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            kl_scheduler.step()
            
            ''' 5. Gather losses '''
            
            bar.set_postfix({'loss': f'{loss.item():.5f}', 'epoch': f'{best_epoch}|{epoch}|{max_epoch}', 'kl_w': f'{kl_weight:.5f}'})
            bar.update(1)
            
            if accelerator.sync_gradients:
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                reduced_eik_loss = accelerator.reduce(eik_loss.detach(), reduction="mean")
                reduced_bbox_loss = accelerator.reduce(bbox_loss.detach(), reduction="mean")
                reduced_kl_loss = accelerator.reduce(kl_loss.detach(), reduction="mean")
                if accelerator.is_main_process:
                    if i % 100 == 0 and exp is not None:
                        exp.log_metrics({
                                            'loss': reduced_loss.item(),
                                            'eik': reduced_eik_loss.item(),
                                            'bbox': reduced_bbox_loss.item(),
                                            'kl_loss': reduced_kl_loss.item(),
                                            'kl_weight': kl_weight,
                                        },
                                        prefix='train',
                                        step = epoch * len(train_dataloader) + i + 1,
                                        epoch=epoch,)


    #     if i > 100:
    #         break
        
    # collector.compose()
    # collector.numpy()
    # collector.save(f'tempfile/{epoch}_{accelerator.process_index}.pkl')
    # print('saved')
    # exit()
    
    
    bar.close()
    
    # if epoch % 3 != 0:
    #     continue
    
    encoder.eval()
    decoder.eval()  
      
    # accelerator.print("\nEvaluating on test set...")
    with torch.no_grad():
        bar = tqdm(test_dataloader, desc='Testing', ascii=True, leave=False, disable=not accelerator.is_main_process)
        for i, data in enumerate(test_dataloader):
            
            if predict_bbox:
                cloud, sample, gt_sdf, uid, gt_joint_loc, gt_joint_axis, gt_bbox_transforms, gt_bbox_sizes = data
                gt_bbox_transforms = gt_bbox_transforms.to(torch.float32)
                gt_bbox_sizes = gt_bbox_sizes.to(torch.float32)
            else:
                cloud, sample, gt_sdf, uid, gt_joint_loc, gt_joint_axis = data
                        
            
            cloud = cloud.to(torch.float32)
            sample = sample.to(torch.float32)
            gt_sdf = gt_sdf.to(torch.float32)
            gt_joint_loc = gt_joint_loc.to(torch.float32)
            gt_joint_axis = gt_joint_axis.to(torch.float32)
            
            
            
            
            # #---test---
            # color_source = normlize(gt_sdf[..., None], dim=-2)
            # rgba = torch.from_numpy(color_map_numpy(color_source.detach().cpu().numpy())).to(device=device)
            # collector.add('cloud', cloud)
            # collector.add('sample1', torch.cat([sample, rgba], dim=-1)[:, 0])
            # #---test---
            gt_sdf = gt_sdf[:, 0]
            sample = sample[:, 0]            
            sample, gt_sdf = random_sample_within_unit_cube(sample, gt_sdf, num_samples=4096)


            num_samples = sample.shape[-2]
            
            
            instance_code = encoder(cloud)
            mu, logvar = instance_code.chunk(2, dim=-1)
            z = mu # Use mean for testing
            
            sdf_meta = decoder(z, sample)
            sdf_pred, norm_joint_loc, norm_joint_axis = sdf_meta['sdf'], sdf_meta['norm_joint_loc'], sdf_meta['norm_joint_axis']

            loss_sdf = lossFunc(torch.clamp(sdf_pred.squeeze(-1),   max=max_dis, min=-max_dis),
                                torch.clamp(gt_sdf,                 max=max_dis, min=-max_dis)
                                ) / num_samples
            loss_sdf = loss_sdf.sum()
            
            kl_loss = compute_kl_loss(mu, logvar) * inscodeRegularizationLambda

            loss_joint = decoder.module.joint_loss(
                norm_joint_loc, norm_joint_axis, gt_joint_loc, gt_joint_axis, get_category_jointtype(category)
            )

            if predict_bbox:
                norm_bbox_center, norm_bbox_size = sdf_meta['norm_bbox_center'], sdf_meta['norm_bbox_size']
                bbox_loss = decoder.module.bbox_loss(
                    norm_bbox_center, norm_bbox_size, gt_bbox_transforms[..., :3, 3], gt_bbox_sizes
                )
            else:
                bbox_loss = torch.tensor([0.], device=accelerator.device, dtype=loss_sdf.dtype)

            gathered_loss_sdf, gathered_loss_joint, gathered_loss_bbox, gathered_kl_loss = accelerator.gather_for_metrics([loss_sdf, loss_joint, bbox_loss, kl_loss])

            if accelerator.is_main_process:
                collector.add('loss', gathered_loss_sdf)
                collector.add('loss_joint', gathered_loss_joint)
                collector.add('loss_bbox', gathered_loss_bbox)
                collector.add('kl_loss', gathered_kl_loss)
            
            bar.update(1)
            
    
        #     if i > 100:
        #         break
        
        # collector.compose()
        # collector.numpy()
        # collector.save(f'tempfile/sdf_test{epoch}.pkl')
        # print('saved')
        # exit()    
    
            
            
            
            
        bar.close()
        
        if accelerator.is_main_process:
        
            collector.compose()
            collector.numpy()
            
            sdf_loss = collector['loss'].mean().item()
            loss_joint = collector['loss_joint'].mean().item()
            bbox_loss = collector['loss_bbox'].mean().item()
            kl_loss = collector['kl_loss'].mean().item()
            all_loss = sdf_loss + loss_joint + bbox_loss + kl_loss
            # print(all_loss)
            
            if exp is not None:
                exp.log_metrics({
                                    'loss': sdf_loss,
                                    'loss_joint': loss_joint,
                                    'loss_bbox': bbox_loss,
                                    'kl_loss': kl_loss,
                                },
                                prefix='test',
                                epoch=epoch,)
                # exp.log_image(
                #     decoder.module.decoder._catcode.data[0].detach().cpu().numpy()
                # )
                
                
            unwrapped_encoder = accelerator.unwrap_model(encoder)
            unwrapped_decoder = accelerator.unwrap_model(decoder)
            params = {
                'encoder': unwrapped_encoder.state_dict(),
                'decoder': unwrapped_decoder.state_dict(),
                'encoder_params': encoder_params,
                'decoder_params': decoder_params,
                'vae': True,
                'epoch': epoch,
                'loss': sdf_loss,
                'catcode_size': catcode_size,
                'inscode_size': inscode_size,
                'num_catcodes': num_catcodes,
                'category': category,
                'category_name': category_name,
                'num_parts': num_parts,
                'joint_type': get_category_jointtype(category),
                'version': VERSION,
                'train_dataset': train_dataset.get_file_path(),
                'test_dataset': test_dataset.get_file_path(),
                'dataset_name': dataset_name,
            }


            if all_loss < best_loss:
                best_loss = all_loss
                best_epoch = epoch

                if not os.path.exists(WORK_DIR):
                    os.makedirs(WORK_DIR)
                torch.save(unwrapped_decoder.state_dict(),  os.path.join(WORK_DIR, f'sdf_decoder_{dataset_name}_best.pth'))
                torch.save(unwrapped_encoder.state_dict(),  os.path.join(WORK_DIR, f'sdf_encoder_{dataset_name}_best.pth'))
                torch.save(params,                          os.path.join(WORK_DIR, f'sdf_misc_{dataset_name}_best.pth'))
                
                print(f'saved new model at {WORK_DIR}, epoch {epoch}, loss {best_loss}')
                
                with open(os.path.join(WORK_DIR, 'Best_metric.md'), 'w') as f:
                    f.write('#### Best Metric \n\n epoch ' + str(epoch) + '\n\n')
                    f.write(f'**Loss:** {best_loss:.6f} \n\n')
                    f.write('\n\n #### Config \n\n')
                    f.write(readable_dict(OmegaConf.to_container(args, resolve=True), markdown=True))

                # visualc = collector.slice(step=50)
                # visualc.save(os.path.join(WORK_DIR, f'visual/{epoch}.pkl'))
            if epoch % 10 == 0:
                # every 10 epochs, save current model
                if not os.path.exists(WORK_DIR):
                    os.makedirs(WORK_DIR)
                torch.save(params, os.path.join(WORK_DIR, f'sdf_misc_{dataset_name}_EPOCH{epoch}.pth'))
                
                accelerator.save_state(os.path.join(WORK_DIR, 'accelerator_state_latest'))
                with open(os.path.join(WORK_DIR, 'accelerator_state_latest', 'training_state.json'), 'w') as f:
                    json.dump({'epoch': epoch + 1}, f)

        
    # exit()

