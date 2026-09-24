import argparse
import os
import time
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import trimesh
from accelerate import Accelerator
from einops import repeat

from backbone.PointComplete.utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from models import import_model_module
from tools.quaternionOps import QuaternionOperations
from tools.rotation import matrix_to_quaternion, quaternion_to_matrix
from tools.select_utils import *
from tools.utils import *

try:
    from backbone.PointComplete.models.PointAttN_PP import Model as Complete_Net
except:
    print('complete_net import failed')
    traceback.print_exc()

warnings.filterwarnings("ignore")


def single_side_chamfer_per_part_by_seg(projected_kp, cloud, part_cls: torch.Tensor, direction=True, logdis=False, scope='part', norm_sample=64, norm_vail_rate=30., score_type='dis'):
    """Compute a per-part one-sided Chamfer score."""
    B, M, P, K, _ = projected_kp.shape
    _, N, _, = cloud.shape

    if scope == 'part':
        start = 1
        end = P
    elif scope == 'base':
        start = 0
        end = 1
    elif scope == 'all':
        start = 0
        end = P
    else:
        raise ValueError(f'Unknown range: {scope}')


    blist = []

    if score_type == 'dis+norm' or score_type == 'norm':

        global_centroid_kp = projected_kp.reshape(B, M, -1, 3).mean(dim=-2, keepdim=True).unsqueeze(-2)
        lra_kp_pp = compute_LRA_all(projected_kp, weighting=True, shared_centroid=global_centroid_kp)
        global_centroid_cloud = cloud.mean(dim=-2, keepdim=True).unsqueeze(-2)
        cloud_pp = torch.stack([cloud]*P, dim=-3)
        cloud_pp = torch.where(part_cls.unsqueeze(-3).expand_as(cloud_pp) == torch.arange(0, P, device=part_cls.device)[None, :, None, None], cloud_pp, torch.nan)
        cloud_pp_centroid = cloud_pp.nanmean(dim=-2, keepdim=True)
        cloud_pp = torch.where(torch.isnan(cloud_pp), cloud_pp_centroid, cloud_pp)
        lra_cloud_pp = compute_LRA_all(cloud_pp, weighting=True, shared_centroid=global_centroid_cloud)
        cos_theta = torch.einsum('...i,...i->...', lra_kp_pp, lra_cloud_pp.unsqueeze(-3))
        if scope == 'part':
            cur_cos_sim = cos_theta[..., 1:].mean(-1).abs()
        elif scope == 'base':
            cur_cos_sim = cos_theta[..., 0].abs()
        elif scope == 'all':
            cur_cos_sim = cos_theta.mean(-1).abs()
        else:
            raise ValueError(f'Unknown scope: {scope}')

    for b in range(B):
        plist = []
        for part_id in range(start, end):
            _1bkp = projected_kp[b, :, part_id, None]
            dis = torch.linalg.norm(_1bkp - cloud[b, None, :, None], dim=-1)
            dis = torch.where(part_cls[b][None] == part_id, dis, torch.inf)


            if direction:
                min_dis_kp, _ = dis.min(dim=1)
                min_dis_kp = torch.nan_to_num(min_dis_kp, posinf=torch.nan)

            else:
                min_dis_kp, _ = dis.min(dim=2)
                min_dis_kp = torch.nan_to_num(min_dis_kp, posinf=torch.nan)

            if logdis:
                min_dis_kp = -torch.log(min_dis_kp)

            if score_type == 'dis':
                score = min_dis_kp.nanmean(-1)
            elif score_type == 'norm':
                score = cur_cos_sim[b]
            elif score_type == 'dis+norm':
                score = min_dis_kp.nanmean(-1) * cur_cos_sim[b]
            else:
                raise ValueError(f'Unknown score type: {score_type}')


            plist.append(score)

        blist.append(torch.stack(plist, dim=-1))
    dis_per_part = torch.stack(blist, dim=0)

    min_dis_per_sample = dis_per_part.mean(-1, keepdim=True)

    return min_dis_per_sample


def pose_mixing(poses, mask=None):
    """Average translations and rotations across pose hypotheses."""
    B, S, P, _, _ = poses.shape

    if mask is None:
        mask = torch.ones(B, S, 1, 1, 1, device=poses.device)
    else:
        mask = torch.where(mask, mask, torch.nan)


    mixed_t = torch.nanmean(poses[..., :3, 3:4] * mask, dim=1, keepdim=True)
    qut = matrix_to_quaternion(poses[..., :3, :3])  # [w, x, y, z]


    ops = QuaternionOperations()
    mixed_qut = ops.compute_mean(qut, dim=1).unsqueeze(1)

    mixed_R = quaternion_to_matrix(mixed_qut)
    mixed_pose = compose_rt(mixed_R, mixed_t)
    return mixed_pose

def vector_mixing(vectors, normlize=False):
    """Average vectors across hypotheses, with optional normalization."""
    mixed_v = torch.mean(vectors, dim=1, keepdim=True)
    if normlize:
        mixed_v = mixed_v / torch.linalg.norm(mixed_v, dim=-1, keepdim=True)
    return mixed_v


def joint_score(gen_joint_loc, gen_joint_axis, reg_joint_loc, reg_joint_axis):
    """Score generated joints against regressed joint parameters."""
    rdiff = batch_angle_between_vectors(gen_joint_axis, reg_joint_axis[:, None])
    tdiff = point_to_line_distance(gen_joint_loc, reg_joint_loc[:, None], reg_joint_axis[:, None])


    return (rdiff + tdiff).mean(-1, keepdim=True)


def get_mesh_from_decoder(decoder, beta, grid_size=64, bounds=(-1.0, 1.0), level=0.0):
    """Reconstruct a mesh and normalized joint parameters from an SDF code."""
    grid_points = create_grid_points_from_bounds(bounds[0], bounds[1], grid_size)
    grid_points = torch.from_numpy(grid_points).to(beta.device, dtype=beta.dtype)
    grid_points = grid_points.view(1, -1, 3)

    sdf_meta = decoder(beta, grid_points, chunk_size=16)
    sdf_pred, norm_joint_loc, norm_joint_axis = sdf_meta['sdf'], sdf_meta['norm_joint_loc'], sdf_meta['norm_joint_axis']
    sdf_values = sdf_pred.squeeze(0).detach().cpu().numpy().reshape(grid_size, grid_size, grid_size)

    verts, faces, _, _ = marching_cubes(sdf_values, level=level)
    verts = verts / (grid_size - 1) * 2 - 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    return mesh, norm_joint_loc, norm_joint_axis


def select_by_joint(pred_pose, gen_joint_loc, gen_joint_axis, reg_joint_loc, reg_joint_axis, norm_part_kp=None, cloud=None, seg=None, bind_part=0, topk=50):
    """Select and aggregate pose hypotheses using joint consistency."""
    project_gen_joint_loc = transform_points(pred_pose[:, :, bind_part], gen_joint_loc)
    project_gen_joint_axis = transform_points_33(pred_pose[:, :, bind_part, :3, :3], gen_joint_axis)

    jscore = joint_score(project_gen_joint_loc, project_gen_joint_axis, reg_joint_loc, reg_joint_axis)


    _, i = torch.topk(-jscore, topk, dim=1)


    if norm_part_kp is not None and cloud is not None and seg is not None:
        projected_kp = transform_points(pred_pose, norm_part_kp)
        norm_score = single_side_chamfer_per_part_by_seg(projected_kp, cloud, seg, direction=True, logdis=True, scope='base', score_type='norm')
        norm_score_candi, _ = select_by_id(i, norm_score)
        _, i_2 = torch.topk(norm_score_candi, topk//2, dim=1)
        i, _ = select_by_id(i_2, i)

    pose_candi, select_id = select_by_id(i, pred_pose)
    joint_loc_candi, _ = select_by_id(i, gen_joint_loc)
    joint_axis_candi, _ = select_by_id(i, gen_joint_axis)

    selected_pose = pose_mixing(pose_candi)
    selected_joint_loc = vector_mixing(joint_loc_candi)
    selected_joint_axis = vector_mixing(joint_axis_candi, normlize=True)

    grafted_pred_pose = pred_pose.clone()
    grafted_pred_pose[:, :, 0:1] = selected_pose[:, :, 0:1]

    return selected_pose, grafted_pred_pose, selected_joint_loc, selected_joint_axis, select_id

fps_list = []

@torch.no_grad()
def eval(args, accelerator: Accelerator, dataloader: torch.utils.data.DataLoader, complete_net=None, sdf_decoder=None, verbose=True, joint_type='revolute'):
    """Evaluate pose hypotheses and save selection metrics."""
    color = torch.tensor([
        [.78, .32, .16],
        [.90, .76, .52],
        [.98, .95, .77],
        [.45, .66, .57],
        [0.0, .52, .52],
        [0.0, 0.0, 0.0],
    ], device=accelerator.device)


    if accelerator.is_main_process:
        collector = MetricCollector(auto_convert=True)

    bar = tqdm.tqdm(total=len(dataloader), desc='Eval', leave=False, disable=not accelerator.is_main_process, ascii=True)

    eval_counter = 0

    for did, data in enumerate(dataloader):


        post_fix = {
            'wo_point': False,
            'wo_sdf': False,
            'wo_joint': False,
            'inf_time': 0.0,
        }


        cloud = data['cloud']
        seg_mask = data['seg_mask']
        gen_prob = data['gen_prob']
        pred_pose = data['pred_pose']
        gt_pose = data['gt_pose']
        gen_joint_loc_unit = data['gen_joint_loc']
        gen_joint_axis_unit = data['gen_joint_axis']
        reg_joint_loc = data['reg_joint_loc']
        reg_joint_axis = data['reg_joint_axis']
        gen_joint_state = data['gen_joint_state']
        gt_joint_state = data['gt_joint_state']
        gt_norm_joint_loc = data['gt_norm_joint_loc']
        gt_norm_joint_axis = data['gt_norm_joint_axis']
        gt_observed_joint_loc = data['gt_observed_joint_loc']
        gt_observed_joint_axis = data['gt_observed_joint_axis']
        beta = data['beta']
        gt_beta = data['gt_beta']
        scale = data['scale'].unsqueeze(-1)  # B, S, 1, 1


        if args.num_samples != 1024:
            gen_prob = gen_prob[:, :args.num_samples]
            pred_pose = pred_pose[:, :args.num_samples]
            gen_joint_loc_unit = gen_joint_loc_unit[:, :args.num_samples]
            gen_joint_axis_unit = gen_joint_axis_unit[:, :args.num_samples]
            gen_joint_state = gen_joint_state[:, :args.num_samples]
            scale = scale[:, :args.num_samples]
            beta = beta[:, :args.num_samples]


        if args.wo_shape:
            _gen_joint_loc_unit, _gen_joint_axis_unit = sdf_decoder.module.forward_joint(torch.zeros_like(beta[0]))
            gen_joint_loc_unit = _gen_joint_loc_unit.unsqueeze(0)
            gen_joint_axis_unit = _gen_joint_axis_unit.unsqueeze(0)
            beta = torch.zeros_like(beta)


        reg_joint_axis = reg_joint_axis / torch.linalg.norm(reg_joint_axis, dim=-1, keepdim=True)


        gen_joint_loc = gen_joint_loc_unit / scale
        gen_joint_axis = gen_joint_axis_unit


        try:
            _, counts = torch.unique(seg_mask, return_counts=True)
            parts_num_points = counts[1:]
            parts_num_points = parts_num_points.min()
            occ_rate = torch.tensor([parts_num_points.float() / cloud.shape[1]], device=accelerator.device, dtype=cloud.dtype)
        except:
            occ_rate = torch.tensor([0.0], device=accelerator.device, dtype=cloud.dtype)


        inf_time = time.time()

        if args.wo_joint:

            selected_joint_loc = gen_joint_loc[:, :1]
            selected_joint_axis = gen_joint_axis[:, :1]
            refine1_pred_pose = pred_pose
            post_fix['wo_joint'] = True
            S = gen_joint_state.shape[1]
            B = pred_pose.shape[0]

        else:

            selected_pose, _, selected_joint_loc, selected_joint_axis, select_id = select_by_joint(pred_pose=pred_pose,
                                                                                                                    gen_joint_loc=gen_joint_loc,
                                                                                                                    gen_joint_axis=gen_joint_axis,
                                                                                                                    reg_joint_loc=reg_joint_loc,
                                                                                                                    reg_joint_axis=reg_joint_axis,
                                                                                                                    bind_part=0,
                                                                                                                    topk=100 if args.num_samples >= 100 else args.num_samples
                                                                                                                    )

            selected_observed_joint_axis = transform_points_33(selected_pose[:, :, 0, :3, :3], selected_joint_axis)
            selected_observed_joint_loc = transform_points(selected_pose[:, :, 0], selected_joint_loc)


            selected_pose, _, selected_joint_loc, selected_joint_axis, select_id = select_by_joint(pred_pose=pred_pose,
                                                                                                                    gen_joint_loc=gen_joint_loc,
                                                                                                                    gen_joint_axis=gen_joint_axis,
                                                                                                                    reg_joint_loc=selected_observed_joint_loc[:, 0],
                                                                                                                    reg_joint_axis=selected_observed_joint_axis[:, 0],

                                                                                                                    bind_part=0,
                                                                                                                    topk=120 if args.num_samples >= 120 else args.num_samples
                                                                                                                    )


            S = gen_joint_state.shape[1]
            B = pred_pose.shape[0]


            refine1_part_pose = base_pem.decode_articulation(selected_pose[:, :, 0].repeat(1, S, 1, 1), gen_joint_loc, gen_joint_axis, gen_joint_state, joint_type=joint_type)
            refine1_pred_pose = torch.cat([selected_pose[:, :, 0:1].repeat(1, S, 1, 1, 1), refine1_part_pose], dim=2)


        if complete_net is None:
            B, N, _ = cloud.shape
            P = pred_pose.shape[2]

            batch_expanded_cloud = torch.zeros((B, P, N, 3), device=cloud.device, dtype=cloud.dtype)

            for b in range(B):
                c_seg = seg_mask[b, :, 0].long()
                c_cloud = cloud[b]

                for p in range(P):
                    indices = (c_seg == p).nonzero(as_tuple=True)[0]

                    if len(indices) > 0:
                        rand_idx = torch.randint(0, len(indices), (N,), device=cloud.device)
                        batch_expanded_cloud[b, p] = c_cloud[indices[rand_idx]]
                    else:
                        rand_idx = torch.randint(0, N, (N,), device=cloud.device)
                        batch_expanded_cloud[b, p] = c_cloud[rand_idx]

            reprojected_cloud = transform_points(torch.linalg.inv(refine1_pred_pose), batch_expanded_cloud.unsqueeze(1))
        else:
            cloud_net_complete = complete_net(cloud, seg_mask.squeeze(-1), only_test=True)  # B, P, N, 3
            reprojected_cloud = transform_points(torch.linalg.inv(refine1_pred_pose), cloud_net_complete.unsqueeze(1))


        reprojected_cloud = reprojected_cloud * scale.unsqueeze(-3)


        cloud_calcu_sdf = reprojected_cloud


        cloud_calcu_sdf = cloud_calcu_sdf[:, :, 1:, ...]
        cloud_calcu_sdf = cloud_calcu_sdf.reshape(B, S, -1, 3).contiguous()  # B, S, (P-1)*N, 3

        cloud_calcu_sdf = fps_batch(cloud_calcu_sdf, 1024)  # B, S, 1024, 3


        assert cloud_calcu_sdf.shape[0] == 1, "Computed SDF only supports batch size 1, but got {}".format(cloud_calcu_sdf.shape[0])

        if not args.wo_sdf:
            sdf_meta = sdf_decoder(beta[0], cloud_calcu_sdf[0], chunk_size=18)
            raw_sdf = sdf_meta['sdf'].unsqueeze(0)  # B, 1024, 1024, 1
            post_fix['wo_point'] = True


        elif args.wo_sdf and not args.wo_point:
            cham_loss = dist_chamfer_3D.chamfer_3DDist()

            zero_mesh, _, _ = get_mesh_from_decoder(sdf_decoder, beta[:, 0], level=args.level)
            cani_points = torch.tensor(zero_mesh.sample(1024), device=cloud_calcu_sdf.device, dtype=cloud_calcu_sdf.dtype).unsqueeze(0)
            _, _d2, _, _ = cham_loss(repeat(cani_points, 'b n c -> (s b) n c', s=cloud_calcu_sdf.shape[1]),
                            cloud_calcu_sdf[0])
            raw_sdf = _d2.mean(1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            post_fix['wo_sdf'] = True


        elif args.wo_sdf and args.wo_point:
            post_fix['wo_sdf'] = True
            post_fix['wo_point'] = True
            raw_sdf = torch.ones((1, cloud_calcu_sdf.shape[1], 1, 1), device=cloud_calcu_sdf.device)
            raw_sdf[0, 0, 0, 0] = 0.1

        else:
            raise NotImplementedError("wo_sdf and wo_point setting not recognized")


        sdf_score = -torch.abs(raw_sdf)


        sdf_score = torch.nanmean(sdf_score, dim=-2)
        _, i = torch.topk(sdf_score, 1, dim=1)


        pose_candi, select_id = select_by_id(i, refine1_pred_pose)
        selected_pose = pose_mixing(pose_candi)
        select_id = select_id[:, 0,]

        end_inf_time = time.time()


        selected_cd = torch.zeros((B, 1), device=accelerator.device)


        rdiff_sel, tdiff_sel = batch_pose_diff(selected_pose, gt_pose.unsqueeze(1))
        rdiff, tdiff = batch_pose_diff(pred_pose, gt_pose.unsqueeze(1))
        rdiff_zero_axis = batch_angle_between_vectors(gen_joint_axis[:, 0], gt_norm_joint_axis)
        tdiff_zero_loc = point_to_line_distance(gen_joint_loc[:, 0], gt_norm_joint_loc, gt_norm_joint_axis)
        rdiff_zero_state = (gen_joint_state[:, 0] - gt_joint_state)[:, 1:].abs() / torch.pi * 180

        zero_joint_score = tdiff_zero_loc + rdiff_zero_axis

        rdiff_axis_reg = batch_angle_between_vectors(reg_joint_axis, transform_points_33(gt_pose[:, 0, :3, :3], gt_norm_joint_axis))
        tdiff_loc_reg = point_to_line_distance(reg_joint_loc, transform_points(gt_pose[:, 0], gt_norm_joint_loc), transform_points_33(gt_pose[:, 0, :3, :3], gt_norm_joint_axis))


        iou_sel = torch.zeros_like(rdiff_sel)
        iou_nocs_sel = torch.zeros_like(rdiff_sel)


        best_pose_id = (rdiff).sum(-2).min(dim=1)[1]
        best_pose, _ = select_by_id(best_pose_id, pred_pose)
        rdiff_best, tdiff_best = batch_pose_diff(best_pose, gt_pose.unsqueeze(1))


        if accelerator.is_main_process and args.visual:
            visual_collect = Collector(auto_convert=True)
            visual_collect.add('cloud_seg', get_colored_cloud(cloud, seg_mask, color))

            visual_collect.add('cloud_calcu_sdf_sel', select_by_id(select_id, cloud_calcu_sdf)[0])
            visual_collect.add('cloud_calcu_sdf_zero', cloud_calcu_sdf[:, 0:1])
            visual_collect.add('cloud_calcu_sdf_best', select_by_id(best_pose_id, cloud_calcu_sdf)[0])

            visual_collect.add('pred_pose', pred_pose)
            visual_collect.add('gt_pose', gt_pose)
            visual_collect.add('select_pose', selected_pose)
            visual_collect.add('gen_axis_line', build_line(gen_joint_loc + gen_joint_axis, gen_joint_loc)[:, :5])
            visual_collect.add('reg_axis_line', build_line(reg_joint_loc + reg_joint_axis, reg_joint_loc))

            visual_collect.add('gt_norm_axis_line', build_line(gt_norm_joint_loc + gt_norm_joint_axis, gt_norm_joint_loc))
            visual_collect.add('gt_observed_axis_line', build_line(gt_observed_joint_loc + gt_observed_joint_axis, gt_observed_joint_loc))

            visual_collect.add('reprojected_cloud', reprojected_cloud[:, 0:1])

            visual_collect.add('select_axis_line', build_line(selected_joint_loc + selected_joint_axis, selected_joint_loc)[:, :5])


            mpoints = prepare_marching_cubes_points(device=accelerator.device)
            mpoints = mpoints.repeat(beta.shape[0], 1, 1)

            sdf_mp = sdf_decoder(select_by_id(select_id, beta)[0][:, 0], mpoints, chunk_size=16)['sdf']

            verts, faces = marching_cubes_from_sdf(sdf_mp, level=0.01)
            sdf_mesh = {
                'vertex': verts[0],
                'face': faces[0],
            }

            sdf_mp_gt = sdf_decoder(gt_beta, mpoints, chunk_size=16)['sdf']
            verts_gt, faces_gt = marching_cubes_from_sdf(sdf_mp_gt, level=0.01)
            sdf_mesh_gt = {
                'vertex': np.concatenate([verts_gt[0], np.array([[0.2, 0.8, 0.2]]).repeat(verts_gt[0].shape[0], axis=0)], axis=-1),
                'face': faces_gt[0],
            }

            visual_collect.compose()
            visual_collect.numpy()
            visual_collect = visual_collect.get_all()
            visual_collect.update({'sdf_mesh': sdf_mesh, 'sdf_mesh_gt': sdf_mesh_gt})
            os.makedirs(os.path.join(WORK_DIR, 'visual_slice'), exist_ok=True)
            with open(os.path.join(WORK_DIR, 'visual_slice', f'visual_{did}.pkl'), 'wb') as f:
                pickle.dump(visual_collect, f)


        data_to_be_gather = {
            'gen_prob': gen_prob,
            'rdiff_zero': rdiff[:, 0, :],
            'tdiff_zero': tdiff[:, 0, :],
            'rdiff_best': rdiff_best,
            'tdiff_best': tdiff_best,
            'rdiff_sele': rdiff_sel[:, 0, :],
            'tdiff_sele': tdiff_sel[:, 0, :],
            'rdiff_axis_reg': rdiff_axis_reg,
            'tdiff_loc_reg': tdiff_loc_reg,
            'iou_sele': iou_sel,
            'iou_nocs_sele': iou_nocs_sel,
            'gt_pose': gt_pose,
            'pred_joint_loc': gen_joint_loc,
            'gt_norm_joint_axis_line': build_line(gt_norm_joint_loc, gt_norm_joint_loc+gt_norm_joint_axis),
            'gt_norm_joint_loc': gt_norm_joint_loc,

            'selected_cd': selected_cd,


            'rdiff_zero_axis': rdiff_zero_axis,
            'tdiff_zero_loc': tdiff_zero_loc,
            'rdiff_zero_state': rdiff_zero_state,
            'zero_joint_score': zero_joint_score,

            'occ_rate': occ_rate,
        }


        gathered_data = accelerator.gather_for_metrics(data_to_be_gather)


        if accelerator.is_main_process:
            collector.add_batch(gathered_data)


        post_fix['inf_time'] = end_inf_time - inf_time
        fps = 1.0 / (end_inf_time - inf_time) if (end_inf_time - inf_time) != 0 else 0
        post_fix['fps'] = round(fps, 2)
        fps_list.append(fps)
        bar.set_postfix(post_fix)
        bar.update(1)


    accelerator.wait_for_everyone()

    if accelerator.is_main_process:

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


        acc__5dg__sele = collector.acc_lambda(func_5dg, x='rdiff_sele', )
        acc__5cm__sele = collector.acc_lambda(func_5cm, x='tdiff_sele', )
        acc_5d5cm_sele = collector.acc_lambda(func_5d5cm, r='rdiff_sele', d='tdiff_sele')
        mean_rot_err_sele = collector.acc_lambda(func_mean, x='rdiff_sele', )
        mean_trans_err_sele = collector.acc_lambda(func_mean, x='tdiff_sele', )

        mean_axis_rot_err_reg = collector.acc_lambda(func_mean, x='rdiff_axis_reg', )
        mean_loc_trans_err_reg = collector.acc_lambda(func_mean, x='tdiff_loc_reg', )


        mean_sele_cd = collector.acc_lambda(func_mean, x='selected_cd', )

        md_table = markdown_table([
            ['Metric',          'Best',                     'Zero',                         'Zero Mean',                        'Selected',                     'Selected Mean',                    'Trend'],
            ['acc_5dg',         fnp(acc__5dg__best),        fnp(acc__5dg__zero),            fnp(acc__5dg__zero.mean()),         fnp(acc__5dg__sele),            fnp(acc__5dg__sele.mean()),         get_trend_arrow(acc__5dg__sele.mean(), acc__5dg__zero.mean(), lower_is_better=False),       ],
            ['acc_5cm',         fnp(acc__5cm__best),        fnp(acc__5cm__zero),            fnp(acc__5cm__zero.mean()),         fnp(acc__5cm__sele),            fnp(acc__5cm__sele.mean()),         get_trend_arrow(acc__5cm__sele.mean(), acc__5cm__zero.mean(), lower_is_better=False),       ],
            ['acc_5d_5cm',      fnp(acc_5d5cm_best),        fnp(acc_5d5cm_zero),            fnp(acc_5d5cm_zero.mean()),         fnp(acc_5d5cm_sele),            fnp(acc_5d5cm_sele.mean()),         get_trend_arrow(acc_5d5cm_sele.mean(), acc_5d5cm_zero.mean(), lower_is_better=False),       ],
            ['mean_rot_err',    fnp(mean_rot_err_best),     fnp(mean_rot_err_zero),         fnp(mean_rot_err_zero.mean()),      fnp(mean_rot_err_sele),         fnp(mean_rot_err_sele.mean()),      get_trend_arrow(mean_rot_err_sele.mean(), mean_rot_err_zero.mean(), lower_is_better=True),     ],
            ['mean_trans_err',  fnp(mean_trans_err_best),   fnp(mean_trans_err_zero),       fnp(mean_trans_err_zero.mean()),    fnp(mean_trans_err_sele),       fnp(mean_trans_err_sele.mean()),    get_trend_arrow(mean_trans_err_sele.mean(), mean_trans_err_zero.mean(), lower_is_better=True), ],
            ['cd',              '',                         '',              '',           fnp(mean_sele_cd),              fnp(mean_sele_cd.mean()),           '',           ],

            ['mean_axis_reg',   '',                         fnp(mean_axis_rot_err_reg),],
            ['mean_loc_reg',    '',                         fnp(mean_loc_trans_err_reg),],
        ])
        print(md_table)

        str_vni = 'valid num of instances: ' + str(eval_counter) + f' ({eval_counter/len(dataloader)*100:.2f}%)'
        str_tni = 'total num of instances: ' + str(len(dataloader))
        print(str_vni)
        print(str_tni)

        print('FPS:', np.mean(np.array(fps_list)))

        with open(os.path.join(args.work_dir,
                               f'Selected_metric{"_wo_point" if args.wo_point else ""}' +
                               f'{"_wo_joint" if args.wo_joint else ""}' +
                               f'{"_wo_cmp" if complete_net is None else ""}' +
                               f'{"_wo_sdf" if args.wo_sdf else ""}' +
                               f'{"_wo_shape" if args.wo_shape else ""}' +
                               f'_k{args.num_samples}' if args.num_samples != 1024 else "" +
                               f'_occ{args.occ}' if args.occ > 0.0 else '' +
                               'new_result' +
                               '.md'
                               ),
                  'w') as f:
            f.write('#### Selected Metric \n\n')
            f.write(md_table)
            f.write('\n\n')
            f.write(str_vni + '\n')
            f.write(str_tni + '\n\n')
            f.write('#### Config \n\n')
            f.write(readable_dict(vars(args), markdown=True))


        collector.save(os.path.join(WORK_DIR,
                                    'eval_result_selected' +
                                    f'{"_wo_point" if args.wo_point else ""}' +
                                    f'{"_wo_joint" if args.wo_joint else ""}' +
                                    f'{"_wo_cmp" if complete_net is None else ""}' +
                                    f'{"_wo_sdf" if args.wo_sdf else ""}' +
                                    f'{"_wo_shape" if args.wo_shape else ""}' +
                                    '.pkl'
                                    ))

        try:
            occ_rates = collector.get('occ_rate')

            if occ_rates is not None:
                rdiff_sele_all = collector.get('rdiff_sele')
                tdiff_sele_all = collector.get('tdiff_sele')

                acc_mask = (rdiff_sele_all < 5.0) & (tdiff_sele_all < 0.05)

                bins = np.arange(0.0, 1.00, 0.02)

                x_ticks = []
                x_labels = []

                mean_rot_errs = []
                mean_trans_errs = []
                acc_5d_5cms = []

                valid_bins = []

                for i in range(len(bins) - 1):
                    lower = bins[0]
                    upper = bins[i+1]

                    mask = (occ_rates >= lower) & (occ_rates < upper)

                    if mask.sum() > 0:
                        x_ticks.append(i)
                        x_labels.append(f"{lower}-{upper}")

                        mean_rot_errs.append(rdiff_sele_all[mask].mean())
                        mean_trans_errs.append(tdiff_sele_all[mask].mean())
                        acc_5d_5cms.append(np.float32(acc_mask[mask]).mean())
                        valid_bins.append(i)

                if len(valid_bins) > 1:
                    fig, ax1 = plt.subplots(figsize=(10, 6))

                    ax1.set_xlabel('Occlusion Rate')
                    ax1.set_ylabel('Error')

                    l1, = ax1.plot(x_ticks, mean_rot_errs, 'r-o', label='Mean Rot Err (deg)')
                    l2, = ax1.plot(x_ticks, mean_trans_errs, 'g-s', label='Mean Trans Err')

                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Accuracy')
                    l3, = ax2.plot(x_ticks, acc_5d_5cms, 'b-^', label='Acc 5d 5cm')
                    ax2.set_ylim(0, 1.05)

                    lines = [l1, l2, l3]
                    ax1.legend(lines, [l.get_label() for l in lines], loc='upper left')

                    ax1.set_xticks(x_ticks)
                    ax1.set_xticklabels(x_labels)

                    plt.title('Metrics vs Occlusion Rate')
                    plt.grid(True)
                    plt.savefig(os.path.join(WORK_DIR, f'metrics_vs_occ.png'))
                    plt.close()
                    print(f"Plot saved to metrics_vs_occ.png")
                else:
                    print(f"Not enough bins for plotting. valid bins: {valid_bins} bins: {bins}")

        except Exception as e:
            print(f"Plotting failed: {e}")
            traceback.print_exc()


if __name__ == '__main__':


    fast_input = {
        'hoi4d':{
            'C3':{
                'work_dir': 'model_dict/ape_hoi4d_Laptop/ape_hoi4d_Laptop_loyal_jabuticaba_4963_model_V09_CTX384JTYrevolute',
                'cmp_model': 'model_dict/cmp_hoi4d_Laptop/PointAttN_PP_cd_C3_22_hoi4d_2025_11_26_16_21/best_cd_t_network.pth',
            },
            'C4':{
                'work_dir': 'model_dict/ape_hoi4d_StorageFurniture/ape_hoi4d_StorageFurniture_original_hostel_4358_model_V09_CTX384JTYrevolute',
                'cmp_model': 'model_dict/cmp_hoi4d_StorageFurniture/PointAttN_PP_cd_C4_22_hoi4d_2025_10_31_21_37/best_cd_t_network.pth',
            },
            'C6':{
                'work_dir': 'model_dict/ape_hoi4d_Safe/ape_hoi4d_Safe_blank_stingray_8753_model_V09_CTX384JTYrevolute',
                'cmp_model': 'model_dict/cmp_hoi4d_Safe/PointAttN_PP_cd_C6_22_hoi4d_2025_11_19_19_55/best_cd_t_network.pth',
            },
            'C14':{
                'work_dir': 'model_dict/ape_hoi4d_TrashCan/ape_hoi4d_TrashCan_amber_archipelago_2_model_V09_CTX384JTYrevolute',
                'cmp_model': 'model_dict/cmp_hoi4d_TrashCan/PointAttN_PP_cd_C14_22_hoi4d_2025_11_27_16_21/best_cd_t_network.pth',
            }
        }
    }


    parser = argparse.ArgumentParser(description='Evaluation script.')
    parser.add_argument('--dataset', type=str, default='hoi4d', help='Dataset name')
    parser.add_argument('--category', type=str, default='', help='Category name')
    parser.add_argument('-w', '--work_dir', type=str, default='auto', help='Model Workspace')
    parser.add_argument('-c', '--cmp_model', type=str, default='auto', help='Point completion model path [best_cd_t_network.pth]')
    parser.add_argument('-l', '--level', type=float, default=0.0, help='Marching cubes level')
    parser.add_argument('-o', '--occ', type=float, default=0.0, help='Occlusion threshold')
    parser.add_argument('-v', '--visual', action='store_true', default=False, help='enable visualization')
    parser.add_argument('-k', '--num_samples', type=int, default=1024, help='number of samples')
    parser.add_argument('--wo_point', action='store_true', default=False, help='disable point selection')
    parser.add_argument('--wo_joint', action='store_true', default=False, help='disable joint selection')
    parser.add_argument('--wo_sdf', action='store_true', default=False, help='disable sdf selection')
    parser.add_argument('--wo_shape', action='store_true', default=False, help='use mean shape for all instances')

    args = parser.parse_args()


    if args.work_dir == 'auto':
        if args.dataset in fast_input.keys() and args.category in fast_input[args.dataset].keys():
            args.work_dir = fast_input[args.dataset][args.category]['work_dir']
            print(f'Auto-filled work_dir: {args.work_dir}')
            if args.cmp_model == 'auto':
                args.cmp_model = fast_input[args.dataset][args.category]['cmp_model']
                print(f'Auto-filled cmp_model: {args.cmp_model}')
        else:
            raise ValueError(f'No auto-filled work_dir for dataset {args.dataset} and category {args.category}. Please provide --work_dir and --cmp_model arguments.')


    accelerator = Accelerator()


    WORK_DIR = args.work_dir
    param_dict = torch.load(os.path.join(WORK_DIR, 'model_best.pth'), map_location='cpu', weights_only=False)


    joint_type = param_dict['category_jointtype']
    num_parts = param_dict['category_parts']
    config = param_dict['config']

    pose_model_version = param_dict['version']
    sdf_param_dict = torch.load(config['sdf']['misc_path'], map_location='cpu', weights_only=False)
    sdf_model_version = sdf_param_dict['version']

    if accelerator.is_main_process:
        print(f'joint_type: {joint_type}, num_parts: {num_parts}, pose_model_version: {pose_model_version}, sdf_model_version: {sdf_model_version}')


    sdf_decoder_statedict = param_dict['sdf_decoder']


    h5dataset = H5Dataset_multi(os.path.join(WORK_DIR, 'eval_result_best.h5'),
                                return_dict=True, allow_missing_keys=True)

    h5dataset.setKeys(
        'cloud', 'seg_mask', 'gen_prob',
        'pred_pose', 'gt_pose', 'scale',
        'gen_joint_loc', 'gen_joint_axis', 'reg_joint_loc', 'reg_joint_axis',
        'gen_joint_state', 'gt_joint_state',
        'gt_norm_joint_loc', 'gt_norm_joint_axis', 'gt_observed_joint_loc', 'gt_observed_joint_axis',
        'beta', 'gt_beta',
    )

    dataloader = torch.utils.data.DataLoader(h5dataset, batch_size=1, shuffle=False, num_workers=8, persistent_workers=True)


    sdf_model_module = import_model_module(sdf_model_version)
    base_pem = import_model_module(pose_model_version).base_pem

    if 'decoder_params' in sdf_param_dict.keys():
        SDF_DEC = sdf_model_module.SDFDecoder(**sdf_param_dict['decoder_params'])
    else:
        SDF_DEC = sdf_model_module.SDFDecoder(
                num_catcodes=sdf_param_dict['num_catcodes'],
                catcode_size=sdf_param_dict['catcode_size'],
                inscode_size=sdf_param_dict['inscode_size'],
                num_parts=sdf_param_dict['num_parts'],
                )


    SDF_DEC.load_state_dict(sdf_decoder_statedict)


    dataloader, SDF_DEC = accelerator.prepare(dataloader, SDF_DEC)
    SDF_DEC.eval()

    if os.path.isfile(args.cmp_model):
        accelerator.print(f'Loading completion model from {args.cmp_model}')
        complete_net = Complete_Net(None, num_parts=sdf_param_dict['num_parts'], r2=2)
        complete_net.load_state_dict(torch.load(args.cmp_model, map_location='cpu', weights_only=False)['net_state_dict'])
        complete_net.eval()
        complete_net = accelerator.prepare(complete_net)
    else:
        accelerator.print(f'No completion model found at {args.cmp_model}, using empty completion net.')
        complete_net = None

    eval(args, accelerator, dataloader, complete_net, SDF_DEC, verbose=True, joint_type=joint_type)
