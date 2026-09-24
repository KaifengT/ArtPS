from models import import_model_module


def evaluate_mesh_reconstruction(args):

    # --- Configuration ---

    WORK_DIR = os.path.dirname(args.file)
    
    if args.dataset == 'auto':
        if 'hoi4d' in WORK_DIR:
            args.dataset = 'hoi4d'
        elif 'sapien' in WORK_DIR:
            args.dataset = 'sapien'
        else:
            raise ValueError(f"Cannot infer dataset from WORK_DIR: {WORK_DIR}")

            
    if args.dataset == 'hoi4d':
        from dataloader.hoi4d_dataset.hoi4d_dataset_arti import get_category_name, get_num_parts
    elif args.dataset == 'sapien':
        from dataloader.sapien_dataset.sapien_dataset import get_category_name, get_num_parts
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}") 

    # --- Load Misc Data and Model Package ---
    misc_data = torch.load(args.file, map_location='cpu', weights_only=False)
    pkg = import_model_module(misc_data['version'])


    if args.dataset_path == 'auto':
        args.dataset_path = misc_data['test_dataset']
    if args.category == 'auto':
        args.category = misc_data['category']
    if args.catcode_size == 'auto':
        args.catcode_size = misc_data['catcode_size']
    if args.inscode_size == 'auto':
        args.inscode_size = misc_data['inscode_size']

    category_name = get_category_name(args.category)
    num_parts = get_num_parts(args.category)
    
    

    # --- Dataset ---
    test_dataset = H5Dataset(args.dataset_path)
    test_dataset.setKeys('cloud', 'sample', 'sdf', 'urdf_id', 'joint_loc', 'joint_axis', 'bbox_transforms', 'bbox_extents')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- Model ---
    if 'encoder_params' in misc_data.keys():
        encoder = pkg.SDFEncoder(**misc_data['encoder_params'])
    else:
        encoder = pkg.SDFEncoder(int(args.inscode_size))
        
    if 'decoder_params' in misc_data.keys():
        decoder = pkg.SDFDecoder(**misc_data['decoder_params'])
    else:
        decoder = pkg.SDFDecoder(num_catcodes=32, catcode_size=int(args.catcode_size), inscode_size=int(args.inscode_size), num_parts=num_parts)

    is_vae = misc_data.get('vae', False)
    if is_vae:
        print('VAE prior is used in this model.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    encoder.load_state_dict(misc_data['encoder'])
    decoder.load_state_dict(misc_data['decoder'])
    encoder = encoder.to(device)
    decoder = decoder.to(device)    
    

    encoder.eval()
    decoder.eval()

    # --- Output Directory ---
    output_dir = os.path.join(WORK_DIR, 'reconstructed_meshes')
    os.makedirs(output_dir, exist_ok=True)

    collector = MetricCollector(True)

    print(f"Starting evaluation... Reconstructed meshes will be saved in {output_dir}")

    
    bar = tqdm(test_dataloader, desc='Evaluating', ascii=True, leave=False)
    for i, data in enumerate(bar):

        # if i > 10: break

        cloud, _, _, uid, gt_joint_loc, gt_joint_axis, gt_bbox_transforms, gt_bbox_extents = go_to_device(data, device)
        cloud = cloud.to(torch.float32)

        with torch.no_grad():
        # --- Get latent code ---
            spec_code = encoder(cloud)
            if is_vae:
                mu, logvar = spec_code.chunk(2, dim=-1)
                spec_code = mu # Just use the mean in distribution for evaluation
                
            # spec_code = torch.ones_like(spec_code, dtype=torch.float32, device=cloud.device)  # Use zero shape prior for evaluation
            # --- Generate mesh from SDF ---
            # Create a grid of points
            grid_size = 64
            grid_points = create_grid_points_from_bounds(-1.0, 1.0, grid_size)
            grid_points = torch.from_numpy(grid_points).to(device, dtype=torch.float32)
            grid_points = grid_points.view(1, -1, 3)
            # spec_code.requires_grad_(True) 
            spec_code = spec_code.detach().requires_grad_(False)
            # Predict SDF on the grid
            sdf_meta = decoder(spec_code, grid_points, chunk_size=16)
            sdf_pred, norm_joint_loc, norm_joint_axis = sdf_meta['sdf'], sdf_meta['norm_joint_loc'], sdf_meta['norm_joint_axis']
            
            
            sdf_values = sdf_pred.squeeze(0).detach().cpu().numpy().reshape(grid_size, grid_size, grid_size)
            norm_joint_loc = norm_joint_loc.detach().cpu().numpy()
            norm_joint_axis = norm_joint_axis.detach().cpu().numpy()
            gt_joint_loc = gt_joint_loc.detach().cpu().numpy()
            gt_joint_axis = gt_joint_axis.detach().cpu().numpy()
            cloud = cloud.detach().cpu().numpy()
            
            
        
        grid_size_for_visual = 8
        grid_points_for_visual = create_grid_points_from_bounds(-0.7, 0.7, grid_size_for_visual)
        grid_points_for_visual = torch.from_numpy(grid_points_for_visual).to(device, dtype=torch.float32)
        grid_points_for_visual = grid_points_for_visual.view(1, -1, 3).requires_grad_(True)
        sdf_meta = decoder(spec_code, grid_points_for_visual, chunk_size=16)
        sdf_pred_for_visual = sdf_meta['sdf']
        
            
        grad = torch.autograd.grad(
            outputs=sdf_pred_for_visual,
            inputs=grid_points_for_visual,
            grad_outputs=torch.ones_like(sdf_pred_for_visual),
            create_graph=True,
            retain_graph=True,
        )[0] # same as grid_points B, N, 3
        grad = grad.detach().cpu().numpy()
        grid_points_for_visual = grid_points_for_visual.detach().cpu().numpy()
            
            

        # Extract mesh using marching cubes
        try:
            verts, faces, _, _ = marching_cubes(sdf_values, level=args.level)
            # Rescale vertices to the original [-0.5, 0.5] range
            verts = verts / (grid_size - 1) * 2 - 1.0
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)

            sdf = igl.signed_distance(cloud[0], np.array(verts), np.array(faces))[0]
            joint_err = np.linalg.norm(norm_joint_loc - gt_joint_loc, axis=-1)
            
            collector.add(str(uid[0].item()), np.abs(sdf[None]))
            collector.add('all', np.abs(sdf[None]))
            collector.add('joint_loc_err', joint_err)
            # Save mesh
            mesh_filename = os.path.join(output_dir, f'mesh_{uid[0].item()}_{i}.obj')
            # mesh.export(mesh_filename)
            norm_grad = grad / np.linalg.norm(grad, axis=-1, keepdims=True) / 2.0 + 0.5
            norm_grad = np.concatenate([norm_grad, np.ones((*norm_grad.shape[:-1], 1))], axis=-1, dtype=norm_grad.dtype)
            if i % 20 == 0:
                pkl_data = {

                    'cloud': cloud,
                    'mesh':{
                        'vertex': verts,
                        'face': faces,
                    },
                    'norm_joint_loc': norm_joint_loc,
                    'norm_joint_axis_line': np.stack([norm_joint_loc, norm_joint_axis + norm_joint_loc], axis=-2),
                    'gt_joint_loc': gt_joint_loc,
                    'gt_joint_axis_line': np.stack([gt_joint_loc, gt_joint_axis + gt_joint_loc], axis=-2),
                    'category': category_name,
                    'grad_line': np.stack([np.concatenate([grid_points_for_visual, norm_grad], axis=-1), 
                                           np.concatenate([grid_points_for_visual + 0.2 * grad, norm_grad], axis=-1), 
                                           ], axis=-2),
                    'grid_points': np.concatenate([grid_points_for_visual, norm_grad], axis=-1),
                }
                
                if sdf_meta.get('norm_bbox_center', None) is not None:
                    norm_bbox_center = sdf_meta['norm_bbox_center'].detach().cpu().numpy()
                    norm_bbox_size = sdf_meta['norm_bbox_size'].detach().cpu().numpy()
                    bbox_temp = np.array([
                        [[-1, -1,  1],
                        [-1,  1,  1],
                        [ 1,  1,  1],
                        [ 1, -1,  1],
                        
                        [-1, -1, -1],
                        [-1,  1, -1],
                        [ 1,  1, -1],
                        [ 1, -1, -1],]
                    ], dtype=norm_bbox_size.dtype) / 2.
                    bbox_temp = repeat(bbox_temp, "B N C -> B P N C", P=norm_bbox_size.shape[1])
                    bbox = bbox_temp * norm_bbox_size[..., None, :]
                    bbox = bbox + norm_bbox_center[..., None, :]
                    pkl_data['norm_bbox'] = bbox
                    
                    gt_norm_bbox_center = gt_bbox_transforms[..., :3, 3].detach().cpu().numpy()
                    gt_norm_bbox_size = gt_bbox_extents.detach().cpu().numpy()
                    gt_bbox = bbox_temp * gt_norm_bbox_size[..., None, :]
                    gt_bbox = gt_bbox + gt_norm_bbox_center[..., None, :]
                    pkl_data['gt_norm_bbox'] = gt_bbox
                

                pkl_filename = os.path.join(output_dir, f'mesh_{uid[0].item()}_{i}.pkl')
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(pkl_data, f)
                
                bar.set_postfix({'saved': mesh_filename})
                
                
                
        except (ValueError, RuntimeError) as e:
            print(f"Could not generate mesh for sample {i} ({uid[0].item()}): {e}")
            continue
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")
            break
        
        torch.cuda.empty_cache()

    collector.compose()
    collector.numpy()
    # B, 2048
    
    print('Model: {}'.format(WORK_DIR))
    print('Category: {}'.format(category_name))
    
    for key in collector.keys():
        mean_metric = collector[key].mean()
        std_metric = collector[key].std()
        print(f'{key}: {mean_metric:.6f} ± {std_metric:.6f}')
    
    print("Evaluation finished.")

def create_grid_points_from_bounds(minimun, maximun, res):
    x = np.linspace(minimun, maximun, res)
    y = np.linspace(minimun, maximun, res)
    z = np.linspace(minimun, maximun, res)
    grid_points = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return grid_points.astype(np.float32)

if __name__ == '__main__':
    import argparse, os
    
    parser = argparse.ArgumentParser(description='Evaluate SDF model and reconstruct meshes.')
    parser.add_argument('--dataset', type=str, default='auto', choices=['hoi4d', 'sapien', 'auto'], help='Dataset name.')
    parser.add_argument('--category', type=str, default='auto', help='Object category.')
    parser.add_argument('--catcode_size', type=str, default='auto', help='Length of the shape code.')
    parser.add_argument('--inscode_size', type=str, default='auto', help='Size of the instance code.')
    parser.add_argument('--dataset_path', type=str, default='auto', help='Path to the test dataset.')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA device index.')
    
    parser.add_argument('-f', '--file', type=str, default='', help='File name to save evaluation results.')
    parser.add_argument('-l', '--level', type=float, default=0.0, help='SDF surface level for mesh extraction.')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import torch
    from torch.utils.data import DataLoader
    from tools.utils import *
    from tqdm import tqdm
    import argparse
    import trimesh
    from skimage.measure import marching_cubes
    import pickle
    import importlib
    import igl
    from einops import repeat
    
    
    evaluate_mesh_reconstruction(args)


