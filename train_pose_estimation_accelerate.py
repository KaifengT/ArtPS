import os, sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

import torch
import torch.optim as optim
import json
import tqdm
import pickle
from termcolor import cprint
import tools.utils as utils
from tools.utils import *
from tools.select_utils import *
from omegaconf import OmegaConf
import importlib
from models import import_model_module
# from models.model_V10 import VERSION, base_pem, base_pem_rel_trans
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

class Container():

    def __init__(self, accelerator:Accelerator, CONFIG=None):
        
        self.accelerator = accelerator        
        
        self.BASE_AUG_METHOD = CONFIG.dataset.base_aug_method
        if CONFIG.dataset.base_aug_method == 'axis':
            self.BASE_AUG_METHOD = utils.base_augmentation_per_axis
        elif CONFIG.dataset.base_aug_method == 'free':
            self.BASE_AUG_METHOD = utils.base_augmentation
        else:
            raise ValueError('Unknown base augmentation method')

        self.DEVICE = self.accelerator.device

        self.CONFIG = CONFIG
        
        model_package = import_model_module(self.CONFIG.model.package)
        model_loader = getattr(model_package, self.CONFIG.model.name)
        model_version = getattr(model_package, 'VERSION')
        
        dataset_package = importlib.import_module(self.CONFIG.dataset.package)
        dataset_loader = getattr(dataset_package, self.CONFIG.dataset.object)
        get_category_name = getattr(dataset_package, 'get_category_name')
        get_num_parts = getattr(dataset_package, 'get_num_parts')
        get_category_jointtype = getattr(dataset_package, 'get_category_jointtype')
        self.DATASET = dataset_loader.get_dataset_name()


        accelerate_set_seed(6788, device_specific=True)
        
        self.CATEGORY_NAME = get_category_name(CONFIG.category)
        self.NUM_PARTS = get_num_parts(CONFIG.category)
        self.JTYPE = get_category_jointtype(CONFIG.category)
        
        
        self.PROJECT_NAME = f'ape_{self.DATASET}_{self.CATEGORY_NAME}'
        PROJECT_NAME_PREFIX = f'{model_version}_CTX{CONFIG.model.glow.context_features}JTY{self.JTYPE[0]}'
        IS_DEBUG = True if sys.gettrace() else False
        
        self.MODEL_ROOT_PATH = './model_dict'
        
        self.accelerator.wait_for_everyone()
        
        self.accelerator.print('-'*50)
        self.accelerator.print(f'Current Category : {self.CATEGORY_NAME}, ID: {CONFIG.category}, Parts: {self.NUM_PARTS}, Joint Type: {self.JTYPE}')
        self.accelerator.print('Project Name     :', self.PROJECT_NAME)
        self.accelerator.print(readable_dict(CONFIG))
        self.accelerator.print('-'*50)

        
        if self.accelerator.is_main_process and input('Continue? (y/n): ') not in ('y', 'Y', ''): 
            raise KeyboardInterrupt()
             
        if not IS_DEBUG and self.accelerator.is_main_process and input('Enable Comet ML Experiment? (y/n): ') == 'y':
            self.EXP, exp_name = get_comet_exp(self.PROJECT_NAME, prefix=PROJECT_NAME_PREFIX)
            self.EXP_FOLDER_PATH = self.PROJECT_NAME + '_' + exp_name
            self.SAVE_PATH = os.path.join(self.MODEL_ROOT_PATH, self.PROJECT_NAME, self.EXP_FOLDER_PATH)
            
            self.EXP.log_parameters(CONFIG)
            
            
        else:
            self.EXP = None
            self.EXP_FOLDER_PATH = self.PROJECT_NAME + '_' + PROJECT_NAME_PREFIX
            self.SAVE_PATH = os.path.join(self.MODEL_ROOT_PATH, self.PROJECT_NAME, self.EXP_FOLDER_PATH)
            
            # if init_from_exist is not None:
            #     self.EXP_FOLDER_PATH = init_from_exist.split('/')[-1]
            #     self.SAVE_PATH = init_from_exist
            
        self.accelerator.wait_for_everyone()

        self.accelerator.print('Project Name     :', self.EXP_FOLDER_PATH)
        self.accelerator.print('Save Path        :', self.SAVE_PATH)


        self.MAX_EPOCH = CONFIG.max_epoch
        
        self.accelerator.print('-'*50)
        
        # --- load SDF model ---
        
        sdf_data = torch.load(CONFIG.sdf.misc_path, map_location='cpu', weights_only=False)
        
        
        sdf_model_module = import_model_module(sdf_data['version'])
        
        
        if 'encoder_params' in sdf_data.keys():
            self.SDF_ENC = sdf_model_module.SDFEncoder(**sdf_data['encoder_params'])
        else:
            self.SDF_ENC = sdf_model_module.SDFEncoder(sdf_data['inscode_size'])
        if 'decoder_params' in sdf_data.keys():
            self.SDF_DEC = sdf_model_module.SDFDecoder(**sdf_data['decoder_params'])
        else:
            self.SDF_DEC = sdf_model_module.SDFDecoder(
                        num_catcodes=sdf_data.get('num_catcodes', 32),
                        catcode_size=sdf_data['catcode_size'],
                        inscode_size=sdf_data['inscode_size'],
                        num_parts=self.NUM_PARTS,
                    )
        
        self.SDF_ENC_is_vae = sdf_data.get('vae', False)

        self.SDF_DEC.load_state_dict(sdf_data['decoder'])
        self.SDF_ENC.load_state_dict(sdf_data['encoder'])

        self.MODEL = model_loader(
                        glow_config=CONFIG.model.glow,
                        mamba_config=CONFIG.model.mamba, 
                        num_parts=self.NUM_PARTS,
                        beta_dim=sdf_data['inscode_size'],
                        use_pretrain=CONFIG.model.use_pretrain,
                        joint_type=self.JTYPE,
                        reverse_on_training=CONFIG.model.reverse_on_training,
                        sdf_decoder=self.SDF_DEC,
                        )


        self.SDF_DEC.eval()
        self.SDF_ENC.eval()
        self.SDF_DEC.requires_grad_(False)
        self.SDF_ENC.requires_grad_(False)


        self.OPTI = optim.Adam(
                    [{'params':self.MODEL.parameters()}, ], lr=CONFIG.lr, betas=(0.9, 0.99)
                )




        self.train_dataset = dataset_loader(root_dir=CONFIG.dataset.data_root,
                                   category=CONFIG.category,
                                   num_points=CONFIG.dataset.num_points,
                                   mode='train', 
                                   sdf_mode=False,
                                   add_noise=False,
                                   use_cache=True,
                                   force_write_cache=False,
                                   cache_dir=CONFIG.dataset.cache_root,
                                   verbose=IS_DEBUG
                                )

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                        batch_size=CONFIG.dataset.train_bs, 
                                                        shuffle=True, 
                                                        num_workers=CONFIG.dataset.num_workers,
                                                        drop_last=True)

        self.test_dataset = dataset_loader(root_dir=CONFIG.dataset.data_root,
                                  category=CONFIG.category,
                                  num_points=CONFIG.dataset.num_points,
                                  mode='val', 
                                  sdf_mode=False,
                                  add_noise=False,
                                  use_cache=True,
                                  force_write_cache=False,
                                  cache_dir=CONFIG.dataset.cache_root,
                                  verbose=IS_DEBUG
                                )
        # NOTE: change batch size to bs instead of 2
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                                                       batch_size=CONFIG.dataset.val_bs, 
                                                       shuffle=False, 
                                                       num_workers=CONFIG.dataset.num_workers,
                                                        )



        self.SCHE = optim.lr_scheduler.OneCycleLR(self.OPTI, steps_per_epoch=len(self.train_loader), epochs=self.MAX_EPOCH+1, pct_start=0.3, max_lr=CONFIG.lr, final_div_factor=0.4)
        
        self.color = torch.tensor([
            # [.66, .66, .56],
            [.68, .28, .36],
            [.35, .65, .22],
            [.12, .14, .65],
            [0.0, .52, .52],
            [0.0, 0.0, 0.0],
        ], device=self.DEVICE)
        
        
        self.start_epoch = 0
                    
            
        self.MODEL, self.SDF_ENC, self.SDF_DEC, self.train_loader, self.test_loader, self.OPTI, self.SCHE = self.accelerator.prepare(
            self.MODEL, self.SDF_ENC, self.SDF_DEC, self.train_loader, self.test_loader, self.OPTI, self.SCHE
        )
        self.MODEL_HELPERS = self.accelerator.unwrap_model(self.MODEL)
        
        resume_path = self.CONFIG.get('resume', None)
        if resume_path is not None and os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, 'accelerator_state_latest')
            self.accelerator.load_state(resume_path)
            
            state_file = os.path.join(resume_path, 'training_state.json')
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.start_epoch = state['epoch']

            self.accelerator.print(f"\033[1;34m Resuming training from state: \033[0m {resume_path} at epoch {self.start_epoch}")
                
                

    def train_epoch(self, epoch):
        self.MODEL.train()
        self.MODEL_HELPERS.sdf_decoder.requires_grad_(False)
        self.SDF_ENC.requires_grad_(False)
        
        bar = tqdm.tqdm(total=len(self.train_loader), desc="Training", position=0, leave=False, ascii=True, disable=not self.accelerator.is_main_process)
        
        for i, data in enumerate(self.train_loader):

            cloud, gt_part_cls, gt_rt, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, limits, gt_complete_cloud, complete_cloud_seg, complete_cloud_per_part, video_name, frameid, connection_matrix = data
            
            connection_matrix = connection_matrix[0]
            
            with self.accelerator.accumulate(self.MODEL):
            
                with torch.no_grad():
                    ''' DATA AUGMENTATION ''' 
                    '''
                    gt_complete_cloud gt_complete_cloud_per_part not augmented
                    '''
                    B, P, _, _ = gt_rt.size()
                    noise = torch.randn_like(cloud) * self.CONFIG.dataset.noise_aug_scale
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
                                            scale_range=self.CONFIG.dataset.shape_aug_scale,
                                            return_scale_ratio=True,
                                            )
                        
                        gt_rt[:, part_id] = scaled_part_rt
                        
                        
                        # prismatic joints need scale joint state
                        if part_id > 0 and self.JTYPE[part_id-1] == 'prismatic':
                            state_scale = torch.linalg.norm(gt_norm_joint_axis[:, part_id-1:part_id] * scale_ratio[:, None, :], dim=-1)
                            gt_joint_state[:, part_id] *= state_scale.squeeze(-1)
                        
                        
                        if part_id > 0:
                            gt_norm_joint_loc[:, part_id-1] = scaled_part_norm_joint_loc
                        
                    # scale cloud

                    scale_transform = torch.eye(4, device=gt_rt.device).unsqueeze(0).expand_as(scaled_part_rt).clone()
                    scale_transform[..., :3, :3] = torch.diag_embed(scale_ratio, dim1=-2, dim2=-1)
                    scale_transform = gt_rt[:, 0] @ scale_transform @ torch.linalg.inv(gt_rt[:, 0])
                    
                    cloud = transform_points(scale_transform, cloud)
                    gt_complete_cloud *= scale_ratio.unsqueeze(-2)
                    '''
                    BASE POSE ARGUMENTATION
                    FIXME WARNING if part not connect to the base, the augmentation will be wrong
                    '''
                    for part_id in range(P):    
                        if part_id == 0:
                            cloud, gt_rt = self.BASE_AUG_METHOD(cloud=cloud, gt_part_rt=gt_rt, **self.CONFIG.dataset.base_aug_param)
                        
                        elif part_id > 0:
                            if self.CONFIG.dataset.get('enable_paug', False):
                                cloud, gt_rt, gt_joint_state = part_augmentation(cloud, gt_rt, part_id, gt_part_cls, gt_norm_joint_loc, gt_norm_joint_axis, gt_joint_state,
                                                                                 min_angle=self.CONFIG.dataset.get('paug_min', limits[:, part_id, 0:1]), 
                                                                                 max_angle=self.CONFIG.dataset.get('paug_max', limits[:, part_id, 1:2]),
                                                                                 # min_angle=0, max_angle=0,
                                                                                 joint_type=self.JTYPE[part_id-1],
                                                                                )
                            
                            
                            
                    # gt_observed_joint_axis = transform_points_33(gt_rt[:, 0, :3, :3], gt_norm_joint_axis)
                    # gt_observed_joint_loc = transform_points(gt_rt[:, 0], gt_norm_joint_loc)
                    
                    def transform_joint(pose: torch.Tensor, joint_loc: torch.Tensor, joint_axis: torch.Tensor, connection_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                        '''
                        transform joints to the observed space by kinematic chain.
                        Args:
                            pose(torch.Tensor): (B, P, 4, 4)
                            joint_loc(torch.Tensor): (B, P-1, 3)
                            joint_axis(torch.Tensor): (B, P-1, 3)
                            connection_matrix(torch.Tensor): (P, P) binary matrix, 1 if connected, 0 if not, M[i, j] = 1 means part i is connected to part j, and part i is the child of part j
                        
                        Returns:
                            transformed_joint_loc: (B, P-1, 3)
                            transformed_joint_axis: (B, P-1, 3)
                        '''
                        B, P, _, _ = pose.shape
                        transformed_joint_loc = torch.zeros_like(joint_loc)
                        transformed_joint_axis = torch.zeros_like(joint_axis)

                        for k in range(P - 1):  # joint k connects part k+1 to its parent
                            child_part = k + 1
                            # find parent of child_part: M[child, parent] = 1
                            parent_mask = connection_matrix[child_part] == 1
                            if parent_mask.any():
                                parent_part = torch.where(parent_mask)[0][0].item()
                            else:
                                parent_part = 0  # fallback to base part

                            # transform joint location by parent's world-space pose
                            transformed_joint_loc[:, k:k+1] = transform_points(pose[:, parent_part], joint_loc[:, k:k+1])
                            # transform joint axis by parent's world-space rotation only
                            transformed_joint_axis[:, k:k+1] = transform_points_33(pose[:, parent_part, :3, :3], joint_axis[:, k:k+1])

                        return transformed_joint_loc, transformed_joint_axis
                        
                    gt_observed_joint_loc, gt_observed_joint_axis = transform_joint(
                        gt_rt, gt_norm_joint_loc, gt_norm_joint_axis, connection_matrix
                    )

                    '''
                    scale cloud to a unit cube range in -1.0 ~ 1.0
                    network needs to learn this scale factor to scale the observed cloud to the unit cube, which is fitted with SDF decoder and encoder
                    '''
                    gt_scale_factor = 1. / gt_complete_cloud.reshape(B, -1).max(-1, keepdim=True)[0] * 0.7 # preserve some margin
                    gt_unit_complete_cloud = gt_complete_cloud * gt_scale_factor.unsqueeze(-2)
                
                
                
                    gt_beta = self.SDF_ENC(gt_unit_complete_cloud)
                    if self.SDF_ENC_is_vae:
                        mu, logvar = gt_beta.chunk(2, dim=-1)
                        gt_beta = mu  # Use mean for training

                
                    #FIXME ABLATION NO SHAPE LEARNING
                    # gt_beta = torch.zeros_like(gt_beta)
                    #FIXME END OF ABLATION
                
                    ''' visualization code, remove before flight'''
                    # c = Collector()  
                    
                    # part_pose = self.MODEL.module.decode_articulation(gt_rt[:, 0], norm_joint_loc=gt_norm_joint_loc, norm_joint_axis=gt_norm_joint_axis, joint_state=gt_joint_state, joint_type=self.MODEL.module.joint_type, connection_matrix=connection_matrix)
                    # pred_pose = torch.cat([gt_rt[:, 0].unsqueeze(-3), part_pose], dim=-3)
                    # pred_joint_loc, pred_joint_axis = self.MODEL.module.sdf_decoder.forward_joint(gt_beta)
                    # pred_joint_loc /= gt_scale_factor.unsqueeze(-2)

                    # normal_cloud = torch.zeros_like(cloud)
                    # for part_id in range(P):
                    #     normal_cloud_ppart = transform_points(torch.linalg.inv(gt_rt[:, part_id]), cloud)
                    #     normal_cloud = torch.where(gt_part_cls.unsqueeze(-1).repeat(1, 1, 3) == part_id, normal_cloud_ppart, normal_cloud)

                    # pred_pose_normal_cloud = torch.zeros_like(cloud)
                    # for part_id in range(P):
                    #     pred_pose_normal_cloud_ppart = transform_points(torch.linalg.inv(pred_pose[:, part_id]), cloud)
                    #     pred_pose_normal_cloud = torch.where(gt_part_cls.unsqueeze(-1).repeat(1, 1, 3) == part_id, pred_pose_normal_cloud_ppart, pred_pose_normal_cloud)
                    
                    # c.add('cloud', get_colored_cloud(cloud, gt_part_cls, self.color)) 
                    # c.add('gt_rt', gt_rt)
                    # c.add('pred_pose', pred_pose)
                    # c.add('reproject_cloud', get_colored_cloud(normal_cloud, gt_part_cls, self.color)) # real size with rest state
                    # c.add('pred_pose_reproject_cloud', get_colored_cloud(pred_pose_normal_cloud, gt_part_cls, self.color)) # real size with predicted pose
                    
                    # c.add('gt_joint_state', gt_joint_state)
                    # c.add('gt_norm_joint_loc', gt_norm_joint_loc)
                    # c.add('gt_norm_joint_axis_line', torch.stack((gt_norm_joint_loc, gt_norm_joint_loc + gt_norm_joint_axis), dim=-2))
                    
                    # c.add('gt_observed_joint_loc', gt_observed_joint_loc)
                    # c.add('gt_observed_joint_axis_line', torch.stack((gt_observed_joint_loc, gt_observed_joint_loc + gt_observed_joint_axis), dim=-2))

                    # c.add('gt_unit_complete_cloud', get_colored_cloud(gt_unit_complete_cloud, complete_cloud_seg, self.color)) # should be in a unit cube
                    
                    # c.add('pred_joint_loc', pred_joint_loc)
                    # c.add('pred_joint_axis_line', torch.stack((pred_joint_loc, pred_joint_loc + pred_joint_axis), dim=-2))
                    
                    # for _ in range(complete_cloud_per_part.shape[1]):
                    #     c.add(f'complete_cloud_per_part_{_}', complete_cloud_per_part[:, _, ...] * scale_ratio.unsqueeze(-2)) # back to original scale
                    
                    # c.compose()
                    # c.add('video_name', video_name)
                    # c.add('frameid', frameid)
                    # c.numpy()
                    # c.save(f'tempfile/{self.DATASET}_{self.CATEGORY_NAME}_{i}_{self.accelerator.process_index}.pkl')
                    # if i > 50:exit()
                    # continue
                    
                    ''' end of visualization code '''
                            
                input_point = cloud
                
                # NOTE: scale 'scale' to small range
                # NOTE: scale 'scale' is embed in func tokenlize_articulation
                pose_token = self.MODEL_HELPERS.tokenlize_articulation_2(gt_rt[:, 0], 
                                                                joint_state=gt_joint_state,
                                                                beta=gt_beta,
                                                                scale=gt_scale_factor,
                                                                cloud=cloud,
                                                                part_cls=gt_part_cls,
                                                                )
                

                
                model_out = self.MODEL(input_point, tgt_distrub=pose_token, num_sample=1, connection_matrix=connection_matrix)
                
                
                loss_cls = self.MODEL_HELPERS.cls_loss(model_out['dense_seg_score'], gt_part_cls, self.CONFIG.weight_cls)
                loss_gen = model_out['loss_gen'] * self.CONFIG.weight_gen
                
                loss_joint_reg = self.MODEL_HELPERS.joint_loss(
                    pred_joint_loc=model_out['joint_loc_reg'], # B, S, 1, 3
                    pred_joint_axis=model_out['joint_axis_reg'],
                    gt_joint_loc=gt_observed_joint_loc,
                    gt_joint_axis=gt_observed_joint_axis,
                    joint_type=self.JTYPE,
                )

                loss = loss_gen + loss_cls + loss_joint_reg

                                

                    
                self.OPTI.zero_grad()
                
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.MODEL.parameters(), 1.0)

                self.OPTI.step()
                self.SCHE.step()

                bar.set_description_str(f"epoch:{epoch}")
                bar.set_postfix_str(f"loss:{loss.item():.5f} | lr:{self.OPTI.param_groups[0]['lr']:.5f}") 
                bar.update(1)
                
                # torch.cuda.empty_cache()
                
                if i % 10 == 0 and self.accelerator.sync_gradients:

                    r_loss_all = self.accelerator.reduce(loss.detach(), reduction='mean')
                    r_loss_cls = self.accelerator.reduce(loss_cls.detach(), reduction='mean')
                    r_loss_loss_joint_reg = self.accelerator.reduce(loss_joint_reg.detach(), reduction='mean')

                    if self.EXP is not None:
                    
                        self.EXP.log_metrics({
                            'loss_all': 20. if r_loss_all.item() > 20 else r_loss_all.item(),
                            'loss_cls': r_loss_cls.item(),
                            'loss_joint_reg': r_loss_loss_joint_reg.item(),

                            'lr': self.OPTI.param_groups[0]['lr'],
                        },
                        prefix='train',
                        step = epoch * len(self.train_loader) + i + 1,
                        epoch=epoch,)
                
                    

  
    def eval_epoch(self, save=True, current_epoch=0, force_save_visual=False):
        self.MODEL.eval()
        self.SDF_ENC.eval()
        
        sample_size = 1024
        
        

        with torch.no_grad():
            
            if self.accelerator.is_main_process:
                collector = MetricCollector(auto_convert=True)
            
            bar = tqdm.tqdm(total=len(self.test_loader), desc="Eval", position=0, leave=False, ascii=True, disable=not self.accelerator.is_main_process)
            for i, data in enumerate(self.test_loader):
                
                
                cloud, gt_part_cls, gt_pose, gt_joint_state, gt_norm_joint_loc, gt_norm_joint_axis, limits, gt_complete_cloud, complete_cloud_seg ,_ ,_ ,_, connection_matrix = data
                connection_matrix = connection_matrix[0]

                gt_observed_joint_axis = transform_points_33(gt_pose[:, 0, :3, :3], gt_norm_joint_axis)
                gt_observed_joint_loc = transform_points(gt_pose[:, 0], gt_norm_joint_loc)
                
                gt_scale_factor = 1. / gt_complete_cloud.reshape(gt_complete_cloud.shape[0], -1).max(-1, keepdim=True)[0] * 0.7 # preserve some margin
                gt_unit_complete_cloud = gt_complete_cloud * gt_scale_factor.unsqueeze(-2)

                
                
                gt_beta = self.SDF_ENC(gt_unit_complete_cloud)
                if self.SDF_ENC_is_vae:
                    mu, logvar = gt_beta.chunk(2, dim=-1)
                    gt_beta = mu  # Use mean for evaluation


                B = cloud.shape[0]
                

                input_point = cloud
                
                model_out = self.MODEL(input_point, num_sample=sample_size, connection_matrix=connection_matrix)
                
            
                    
                pred_pose = model_out['pred_pose']
                joint_loc_gen = model_out['joint_loc']
                joint_axis_gen = model_out['joint_axis']
                joint_state_gen = model_out['joint_state']
                
                beta = model_out['beta']
                
                # NOTE: scale back to original scale
                scale = model_out['scale']

                joint_loc_reg = model_out['joint_loc_reg']
                joint_axis_reg = model_out['joint_axis_reg']
                
                
                dense_seg_score = model_out['dense_seg_score']
                seg_mask = torch.max(dense_seg_score, dim=-1)[1].unsqueeze(-1) # B, N
                                                
                
                gen_prob = torch.exp(model_out['log_p'][:, None])
                

                
                
                
                
                
                
                # calculate pose error
                
                joint_axis_reg = joint_axis_reg / torch.linalg.norm(joint_axis_reg, dim=-1, keepdim=True)
                
                rdiff, tdiff = batch_pose_diff(pred_pose, gt_pose.unsqueeze(1))
                

                rdiff_axis_reg = batch_angle_between_vectors(joint_axis_reg, gt_observed_joint_axis) % 360.
                tdiff_loc_reg = point_to_line_distance(joint_loc_reg, gt_observed_joint_loc, gt_observed_joint_axis)


                            
                
                rdiff_axis_gen = batch_angle_between_vectors(joint_axis_gen[:, 0], gt_norm_joint_axis) % 360.
                tdiff_loc_gen = point_to_line_distance(joint_loc_gen[:, 0], gt_norm_joint_loc, gt_norm_joint_axis)
                rdiff_state_gen = ((joint_state_gen - gt_joint_state.unsqueeze(1).expand_as(joint_state_gen))[..., 1:].abs() / torch.pi * 180) % 360.
                
                
                
                            
                # save result
                data_to_be_gathered = {
                    'cloud': cloud,
                    'part_cls': gt_part_cls,
                    'gt_cloud_complete': gt_complete_cloud,
                    'gt_cloud_complete_seg': complete_cloud_seg,

                    'seg_mask': seg_mask,

                    'gen_prob': gen_prob,

                
                    'rdiff_zero': rdiff[:, 0, :],
                    'tdiff_zero': tdiff[:, 0, :],
                
                    'rdiff_best': rdiff.min(dim=1)[0],
                    'tdiff_best': tdiff.min(dim=1)[0],
                
                
                    'pred_pose': pred_pose,
                    'gt_pose': gt_pose,
                
                    'beta': beta,
                    'gt_beta': gt_beta,

                    'scale': scale,
                    'gt_scale': gt_scale_factor,

                    'gen_joint_loc': joint_loc_gen,
                    'gen_joint_axis': joint_axis_gen,
                    'gen_joint_state': joint_state_gen,
                    'gen_joint_axis_line': build_line(joint_loc_gen, joint_loc_gen+joint_axis_gen),
                
                    'gt_norm_joint_axis_line': build_line(gt_norm_joint_loc, gt_norm_joint_loc+gt_norm_joint_axis),
                    'gt_norm_joint_loc': gt_norm_joint_loc,
                    'gt_norm_joint_axis': gt_norm_joint_axis,
                    'gt_joint_state': gt_joint_state,
                
                    'gt_observed_joint_axis_line': build_line(gt_observed_joint_loc, gt_observed_joint_loc+gt_observed_joint_axis),
                    'gt_observed_joint_loc': gt_observed_joint_loc,
                    'gt_observed_joint_axis': gt_observed_joint_axis,
                
                    'reg_joint_loc': joint_loc_reg,
                    'reg_joint_axis': joint_axis_reg,
                    'reg_joint_axis_line': build_line(joint_loc_reg, joint_loc_reg+joint_axis_reg),



                    'tdiff_joint_loc_zero': tdiff_loc_gen,
                    'rdiff_joint_axis_zero': rdiff_axis_gen,
                    'rdiff_state_zero': rdiff_state_gen[:, 0],

                    'tdiff_joint_loc_reg': tdiff_loc_reg,
                    'rdiff_joint_axis_reg': rdiff_axis_reg,
                    'connection_matrix': connection_matrix,
                }
                
                gathered_data = accelerator.gather_for_metrics(data_to_be_gathered)
                
                if self.accelerator.is_main_process:
                    collector.add_batch(gathered_data)
                
                #FIXME
                # if i > 5:
                #     break
                
                bar.update(1)
            bar.close()
            
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_main_process:
            
                collector.compose()
                collector.numpy()
                
                
                
                
                
                func_5dg = lambda x: (x < 5).astype(np.float32)
                func_5cm = lambda x: (x < 0.05).astype(np.float32)
                func_5d5cm = lambda r, d: ((r < 5) * (d < 0.05)).astype(np.float32)
                func_mean = lambda x: x
                
                acc__5dg__zero = collector.acc_lambda(func_5dg, x='rdiff_zero', )
                acc__5dg__best = collector.acc_lambda(func_5dg, x='rdiff_best', )
                acc__5cm__zero = collector.acc_lambda(func_5cm, x='tdiff_zero', )
                acc__5cm__best = collector.acc_lambda(func_5cm, x='tdiff_best', )
                
                acc_5d5cm_zero = collector.acc_lambda(func_5d5cm, r='rdiff_zero', d='tdiff_zero')
                acc_5d5cm_best = collector.acc_lambda(func_5d5cm, r='rdiff_best', d='tdiff_best')
                
                
                

                mean_rot_err_zero = collector.acc_lambda(func_mean, x='rdiff_zero', )
                mean_trans_err_zero = collector.acc_lambda(func_mean, x='tdiff_zero', )
                
                mean_rot_err_best = collector.acc_lambda(func_mean, x='rdiff_best', )
                mean_trans_err_best = collector.acc_lambda(func_mean, x='tdiff_best', )
                


                mean_state_err_zero     = collector.acc_lambda(func_mean, x='rdiff_state_zero', )            
                mean_axis_rot_err_zero = collector.acc_lambda(func_mean, x='rdiff_joint_axis_zero', )
                mean_loc_trans_err_zero = collector.acc_lambda(func_mean, x='tdiff_joint_loc_zero', )
                
                
                mean_axis_rot_reg = collector.acc_lambda(func_mean, x='rdiff_joint_axis_reg', )
                mean_loc_trans_err_reg = collector.acc_lambda(func_mean, x='tdiff_joint_loc_reg', )

                
                md_table = markdown_table([
                    ['Metric',          'Zero',                     'Best',                     ],
                    ['acc_5dg',         fnp(acc__5dg__zero),        fnp(acc__5dg__best),      ],
                    ['acc_5cm',         fnp(acc__5cm__zero),        fnp(acc__5cm__best),      ],
                    ['acc_5d_5cm',      fnp(acc_5d5cm_zero),        fnp(acc_5d5cm_best),      ],
                    ['mean_rot_err',    fnp(mean_rot_err_zero),     fnp(mean_rot_err_best),   ],
                    ['mean_trans_err',  fnp(mean_trans_err_zero),   fnp(mean_trans_err_best), ],
                    ['mean_state_gz',   fnp(mean_state_err_zero), '', '',],
                    ['mean_axis_gz',    fnp(mean_axis_rot_err_zero), '', ],
                    ['mean_loc_gz',     fnp(mean_loc_trans_err_zero), '', ],
                    ['mean_axis_reg',   fnp(mean_axis_rot_reg), '', ],
                    ['mean_loc_reg',    fnp(mean_loc_trans_err_reg), '', ],
                ])
                print(md_table)
                
                
                if self.EXP is not None:
                    for part_id in range(self.NUM_PARTS):
                        self.EXP.log_metrics({
                            'mean_trans':mean_trans_err_zero[part_id],
                            'mean_rot': mean_rot_err_zero[part_id],
                            },
                                        prefix=f'part{part_id}',
                                        epoch=current_epoch,
                        )

                    self.EXP.log_metrics({
                        'state_gen_error':mean_state_err_zero.mean(),
                        'loc_gen_error':min(mean_loc_trans_err_zero.mean(), 10.),
                        'axis_gen_error':mean_axis_rot_err_zero.mean(),
                        
                        'axis_reg_error':mean_axis_rot_reg.mean(),
                        'loc_reg_error':min(mean_loc_trans_err_reg.mean(), 10.),
                        },
                                        
                                        epoch=current_epoch,
                        )
                unwrapped_model = self.accelerator.unwrap_model(self.MODEL)
                unwrapped_sdf_dec = self.accelerator.unwrap_model(self.SDF_DEC)
                unwrapped_sdf_enc = self.accelerator.unwrap_model(self.SDF_ENC)
                
                param_to_save = {
                    'model': unwrapped_model.state_dict(),
                    # 'optimizer': self.OPTI.state_dict(),
                    # 'scheduler': self.SCHE.state_dict(),
                    'epoch': current_epoch,
                    'rdiff': collector[f'rdiff_zero'],
                    'tdiff': collector[f'tdiff_zero'],
                    'acc_5d': acc__5dg__zero,
                    'acc_5d_5cm': acc_5d5cm_zero,
                    'mean_rot_err_zero': mean_rot_err_zero,
                    'main_metric': mean_axis_rot_reg.mean() + mean_rot_err_zero.mean(),
                    'gen_rot_metric': mean_rot_err_zero.mean(),
                    'reg_rot_metric': mean_axis_rot_reg.mean(),
                    'version':unwrapped_model.version,
                    'config': self.CONFIG,
                    'path': self.SAVE_PATH,
                    'sdf_encoder': unwrapped_sdf_enc.state_dict(),
                    'sdf_decoder': unwrapped_sdf_dec.state_dict(),
                    'category': self.CONFIG.category,
                    'category_name': self.CATEGORY_NAME,
                    'category_parts': self.NUM_PARTS,
                    'category_jointtype': self.JTYPE,
                }
                slice_step = 20
                
                
                # rgba = color_map_numpy(collector['ob_kp_prob'], cmap='plasma')
                visual_collector = collector.subframe(keys=[
                    'cloud', 'seg_mask', 'pred_pose', 'gt_pose', 
                    'gen_joint_loc', 'gen_joint_axis_line',
                    'gt_norm_joint_axis_line', 'gt_norm_joint_loc', 'reg_joint_loc', 'reg_joint_axis_line', 'gt_observed_joint_axis_line',
                ])
                
                
                visual_dict = visual_collector.get_all()
                visual_dict['cloud'] = get_colored_cloud_numpy(visual_dict['cloud'], visual_dict['seg_mask'], self.color.cpu().numpy())
                
                if save:
                    if os.path.exists(os.path.join(self.SAVE_PATH, 'model_best.pth')):
                        
                        # main metric
                        dict_old = torch.load(os.path.join(self.SAVE_PATH, 'model_best.pth'), weights_only=False)
                        if is_new_better(dict_old, param_to_save, 'main_metric', smaller_better=True):
                            print(f"Save main metric better model at epoch {current_epoch}")
                            torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_best.pth'))
                            save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_best', subdir='')
                            collector.saveh5(os.path.join(self.SAVE_PATH, f'eval_result_best.h5'))
                            self._save_md_table(md_table, current_epoch, title='Best')

                        # gen metric
                        dict_old = torch.load(os.path.join(self.SAVE_PATH, 'model_gen.pth'), weights_only=False)
                        if is_new_better(dict_old, param_to_save, 'gen_rot_metric', smaller_better=True):
                            print(f"Save gen metric better model at epoch {current_epoch}")
                            torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_gen.pth'))
                            # save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_gen', subdir='')
                            # collector.saveh5(os.path.join(self.SAVE_PATH, f'eval_result_gen.h5'))

                        # reg metric
                        dict_old = torch.load(os.path.join(self.SAVE_PATH, 'model_reg.pth'), weights_only=False)
                        if is_new_better(dict_old, param_to_save, 'reg_rot_metric', smaller_better=True):
                            print(f"Save reg metric better model at epoch {current_epoch}")
                            torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_reg.pth'))
                            # save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_reg', subdir='')
                            # collector.saveh5(os.path.join(self.SAVE_PATH, f'eval_result_reg.h5'))


                        
                    else:
                        os.makedirs(self.SAVE_PATH, exist_ok=True)
                        torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_best.pth'))
                        torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_gen.pth'))
                        torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_reg.pth'))
                        save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_best', subdir='')
                        
                    torch.save(param_to_save, os.path.join(self.SAVE_PATH, 'model_latest.pth'))
                    # save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_latest', subdir='')
                    
                    self.accelerator.save_state(os.path.join(self.SAVE_PATH, 'accelerator_state_latest'))
                    with open(os.path.join(self.SAVE_PATH, 'accelerator_state_latest', 'training_state.json'), 'w') as f:
                        json.dump({'epoch': current_epoch + 1}, f)
                        
                    self._save_md_table(md_table, current_epoch, title='Current')

                    
                if force_save_visual:
                    print('Save eval Result at ', self.SAVE_PATH)
                    os.makedirs(self.SAVE_PATH, exist_ok=True)
                    collector.saveh5(os.path.join(self.SAVE_PATH, f'eval_result_manual.h5'))
                    save_visual_result(visual_dict, self.SAVE_PATH, filename='visual_manual', subdir='')
                
            

    def train(self):
        eval_interval = self.CONFIG.get('eval_interval', 3)
        for epoch in range(self.start_epoch, self.CONFIG.max_epoch):
            self.train_epoch(epoch)
            
            if epoch % eval_interval == 0:
                self.eval_epoch(save=True, current_epoch=epoch)
                
                torch.cuda.empty_cache()
                
    def eval(self):
        self.eval_epoch(save=True, current_epoch=0, force_save_visual=True)
        
        
    def _save_md_table(self, md_table, epoch, title='Best'):
        with open(os.path.join(self.SAVE_PATH, f'{title}_metric.md'), 'w') as f:
            f.write(f'#### {title} Metric \n\n epoch ' + str(epoch) + '\n\n')
            f.write(md_table)
            f.write('\n\n #### Config \n\n')
            f.write(readable_dict(self.CONFIG, markdown=True))


    


if __name__ == "__main__":
    
    '''
    For resume training, add command line argument:
      resume='path_to_accelerator_state_dir'
    
    '''
    
    
    accelerator = Accelerator()
    base_config_path = 'configs/pose/config_pose_base.yaml'
    bCONFIG = OmegaConf.load(base_config_path)
    
    overrides = OmegaConf.from_cli(sys.argv[1:])
    accelerator.print('Overrides from command line:', overrides)
    if 'config' in overrides.keys():
        spec_config_path = overrides.config
        del overrides['config']
        sCONFIG = OmegaConf.load(spec_config_path)
        CONFIG = OmegaConf.merge(bCONFIG, sCONFIG, overrides)

    else:
        raise ValueError('Please provide [config xxx.yaml]')
    

    container = Container(accelerator, CONFIG)

    # NOTE: choose train or eval
    container.train()
    # container.eval()


