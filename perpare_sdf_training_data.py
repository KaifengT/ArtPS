from tools.utils import *
from tqdm import tqdm
# from tqdm.rich import tqdm
import multiprocessing
import igl
import trimesh
import time
from copy import deepcopy
import argparse
from einops import rearrange, repeat
import numpy as np

def sample_surface(
    vertices,
    faces,
    area_faces,
    count,    
    random_generator,
):
    """
    Sample the surface of a mesh, returning the specified
    number of points

    For individual triangle sampling uses this method:
    http://mathworld.wolfram.com/TrianglePointPicking.html

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Geometry to sample the surface of
    count : int
      Number of points to return
    face_weight : None or len(mesh.faces) float
      Weight faces by a factor other than face area.
      If None will be the same as face_weight=mesh.area
    sample_color : bool
      Option to calculate the color of the sampled points.
      Default is False.
    seed : None or int
      If passed as an integer will provide deterministic results
      otherwise pulls the seed from operating system entropy.

    Returns
    ---------
    samples : (count, 3) float
      Points in space on the surface of mesh
    face_index : (count,) int
      Indices of faces for each sampled point
    colors : (count, 4) float
      Colors of each sampled point
      Returns only when the sample_color is True
    """

    
    # len(mesh.faces) float, array of the areas
    # of each face of the mesh
    face_weight = area_faces

    # cumulative sum of weights (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)

    # seed the random number generator as requested
    # if seed is None:
    #     random = np.random.random
    # else:
    #     random = np.random.default_rng(seed).random

    # last value of cumulative sum is total summed weight/area
    face_pick = random_generator(count) * weight_cum[-1]
    # get the index of the selected faces
    face_index = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[faces[:, 0]]
    tri_vectors = vertices[faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = random_generator((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples, face_index




# def generate_uniform_cube_points(resolution=100, min_val=-1, max_val=1):
    
#     coords = np.linspace(min_val, max_val, resolution)
#     x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
#     points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
#     return points


def generate_uniform_cube_points(min_val, max_val, resolution=100, ):
    '''
    Args:
        min_val: 3,
        max_val: 3,
        resolution: int
    '''
    
    xcoords = np.linspace(min_val[0], max_val[0], resolution)
    ycoords = np.linspace(min_val[1], max_val[1], resolution)
    zcoords = np.linspace(min_val[2], max_val[2], resolution)
    
    x, y, z = np.meshgrid(xcoords, ycoords, zcoords, indexing='ij')
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    return points

def process_data(data, num_sample=2048, sigma=0.05, scale_range=0.05, visual=False, is_scale_unit=True):
    
    try:
        
        
        seed = int(time.time() * 1000) % (2**32) + os.getpid()
        
        # if seed is None:
        #     random = np.random.random
        # else:
        random_generator = np.random.default_rng(seed)

        
        
        mesh_list = data['mesh_list']
        # scale_factor = data['scale_factor']
        resampled_points = []
        sdf_list = []
        # resampled_normals = []
                
        for mesh in mesh_list:
            
            if is_scale_unit:
                aug_scale_factor = random_generator.uniform(1-scale_range, 1+scale_range, size=(1, 3))
                scale_factor = (1. / np.abs((data['cloud'] * aug_scale_factor)).max() * 0.7)
                scale_factor = scale_factor * aug_scale_factor
            else:
                scale_factor = random_generator.uniform(1-scale_range, 1+scale_range, size=(1, 3))
                    
            # print('scale_factor', scale_factor)
            F = mesh['faces']
            V = mesh['vertices'] * scale_factor
            
            data['cloud'] *= scale_factor
            
            
            data['joint_loc'] *= scale_factor
            data['joint_axis'] *= scale_factor
            data['bbox_extents'] *= scale_factor
            data['bbox_transforms'][..., :3, 3] *= scale_factor
            # normalize joint axis
            data['joint_axis'] = data['joint_axis'] / (np.linalg.norm(data['joint_axis'], axis=-1, keepdims=True) + 1e-8)
            
            ''' sample func 1 '''
            A = igl.doublearea(V, F) / 2.0
            
            rp = sample_surface(V, F, A, num_sample, random_generator=random_generator.random)[0]
            # rp += sigma * random_generator.randn(num_sample, 3)
            rp += sigma * random_generator.standard_normal(size=rp.shape)
            
            cube_points = random_generator.uniform(-1.3, 1.3, size=(num_sample, 3))
            
            rp = np.concatenate([rp, cube_points], axis=0)
            ind = random_generator.choice(rp.shape[0], num_sample, replace=False)
            rp = rp[ind]
            
            resampled_points.append(rp)
            sdf = igl.signed_distance(np.array(rp), V, F)[0]
            sdf_list.append(sdf)
            
            
            
        
            
        data['sdf'] = np.stack(sdf_list, axis=0, dtype=np.float32)
        data['sample'] = np.stack(resampled_points, axis=0, dtype=np.float16)
        
        # data['cloud16'] = np.float16(data['cloud'])
        # data['sample16'] = np.float16(data['sample'])
        
        if visual:
            color_source = data['sdf'][..., None]
            color_source = color_source / np.max(color_source, keepdims=True)
            rgba = color_map_numpy(color_source)
            rgba = rgba.astype(data['sample'].dtype)
            
            bbox_temp = np.array([
                [[-1, -1,  1],
                 [-1,  1,  1],
                 [ 1,  1,  1],
                 [ 1, -1,  1],
                
                 [-1, -1, -1],
                 [-1,  1, -1],
                 [ 1,  1, -1],
                 [ 1, -1, -1],]
            ], dtype=data['bbox_extents'].dtype) / 2.
            bbox_temp = repeat(bbox_temp, "P N C -> (np P) N C", np=data['bbox_extents'].shape[0])
            bbox = bbox_temp * data['bbox_extents'][:, None]
            bbox = bbox @ data['bbox_transforms'][..., :3, :3].transpose(0, 2, 1) + data['bbox_transforms'][..., :3, 3][:, None, :]
            
            data['sample'] = np.concatenate([data['sample'], rgba], axis=-1)
            data['bbox'] = bbox
        
        del data['mesh_list']
        
        
        for k, v in data.items():
            if v is None:
                raise ValueError(f"Data {k} is None, please check your data.")
        

        return data
    except Exception as e:
        print(f"Error processing data: {e}")
        return {}


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Generate SDF data for HOI4D dataset.')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['sapien', 'hoi4d',], help='Dataset name.')
    parser.add_argument('-c', '--category', type=str, default='C1', help='Object category.')
    parser.add_argument('-v', '--is_visual', action='store_true', default=False, help='Whether to visualize the data.')
    parser.add_argument('-n', '--num_samples', type=int, default=16384, help='Size of the instance code.')
    parser.add_argument('--sigma', type=float, default=0.03, help='Standard deviation for noise.')
    parser.add_argument('--scale_range', type=float, default=0.2, help='Range for scale augmentation.')
    parser.add_argument('--scale_unit', type=int, default=1, help='Whether to scale uniformly across all axes.')
    parser.add_argument('--dataset_len', default='auto', help='Length of the dataset.')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'val'], default='train', help='Mode of the dataset (train or val).')
    
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Path to save the processed data.')
    
    parser.add_argument('--sapien_dataset_root', type=str, default='', help='Root path of the dataset.')
    parser.add_argument('--sapien_dataset_cache', type=str, default='', help='Cache path of the dataset.')
    parser.add_argument('--hoi4d_dataset_root', type=str, default='', help='Root path of the HOI4D dataset.')
    parser.add_argument('--hoi4d_dataset_cache', type=str, default='', help='Cache path of the HOI4D dataset.')
    
    args = parser.parse_args()



    np.random.seed(67883)

    collector = Collector(auto_convert=True)
    
    is_debug = True if sys.gettrace() else False
    is_save = True if not is_debug else False
    is_visual = args.is_visual
    
    if is_visual:
        print('*' * 50)
        print('\n')
        print("WARNING Visual mode is enabled, saving data as pkl.")
        print('\n')
        print('*' * 50)
  

    
    num_points = 2048
    sample_points = args.num_samples
    sigma = args.sigma
    out_path = args.output_path
    mode = args.mode
    scale_range = args.scale_range
    category = args.category
    is_scale_unit = args.scale_unit

    assert mode in ['train', 'val'], "mode should be train or val"
    
    print('Parameters:')
    
    print(readable_dict(vars(args)))
    
    
    if args.dataset_len == 'auto':
        if is_debug or is_visual:
            dataset_len = 100
        elif mode == 'train':
            dataset_len = 300000
        elif mode == 'val':
            dataset_len = 6000
    else:
        dataset_len = int(args.dataset_len)
    
    
    if args.dataset == 'sapien':
        from dataloader.sapien_dataset.sapien_dataset import SapienDataset, get_category_name
        target_dataset = SapienDataset(args.sapien_dataset_root, category=category, mode=mode, sdf_mode=True, sdf_sample_num=dataset_len, 
                                        add_noise= mode=='train',
                                        use_cache=True,
                                        cache_dir=args.sapien_dataset_cache)
    elif args.dataset == 'hoi4d':
        from dataloader.hoi4d_dataset.hoi4d_dataset_arti import HOI4D, get_category_name
        target_dataset = HOI4D(args.hoi4d_dataset_root, category=category, mode=mode, sdf_mode=True, sdf_sample_num=dataset_len, 
                                        add_noise= mode=='train',
                                        use_cache=True,
                                        cache_dir=args.hoi4d_dataset_cache)
    out_path = os.path.join(out_path, f'{target_dataset.get_dataset_name(upper=True)}_SDF')


    max_process = 1 if is_debug else None
    print('Number of process:', max_process)
    print('Starting data processing...')    
    

    bar_queue = tqdm(target_dataset, desc='Processing', ascii=True) 
    bar_finish = tqdm(target_dataset, desc='finished', ascii=True) 

    write_count = 0

    # callback function to collect results
    def collect_result(result:dict):
        for key, value in result.items():
            collector.add(key, value[None])
        bar_finish.update(1)
        
    def collect_and_write_h5(result:dict, f:h5py.File):
        global write_count
        g = f.require_group(str(write_count))
        for key in result.keys():
            g.create_dataset(key, data=result[key])
        write_count += 1

        bar_finish.update(1)
        bar_finish.set_postfix({'written': write_count})
        

    instance_count = {}
    
    try:
        if is_visual or not is_save:
            collector_func = collect_result
            
        else:
            import h5py
            
            folder = os.path.join(out_path, target_dataset.category_name)
            os.makedirs(folder, exist_ok=True)
            _fpath = os.path.join(folder, f'{mode}_sdf_{target_dataset.get_dataset_name()}_{target_dataset.category_name}_p{sample_points}_s{sigma}_a{scale_range}_l{dataset_len}_u{is_scale_unit}_onepart.h5')
            f = h5py.File(_fpath, 'w', track_order=False)
            collector_func = lambda res: collect_and_write_h5(res, f)
        
        
        with multiprocessing.Pool(max_process, ) as pool:

            for i, data in enumerate(target_dataset):
                cloud, mesh_list, part_cls, joint_loc, joint_axis, bbox_extents, bbox_transforms, scale_factor, urdf_id = data
                
                param = {
                    'cloud': cloud,
                    'mesh_list': mesh_list,
                    'part_cls': part_cls,
                    'joint_loc': joint_loc,
                    'joint_axis': joint_axis,
                    'bbox_extents': bbox_extents,
                    'bbox_transforms': bbox_transforms,
                    'scale_factor': scale_factor,
                    'urdf_id': urdf_id
                }
                
                _current_ins_num = instance_count.get(f'{urdf_id:0>2d}', 0)
                instance_count[f'{urdf_id:0>2d}'] = _current_ins_num + 1
                
                detached_param = deepcopy(param)

                pool.apply_async(
                    process_data,
                    args=(detached_param, sample_points, sigma, scale_range, is_visual, is_scale_unit),
                    callback=collector_func
                )
                
                del detached_param
                
                bar_queue.update(1)
            
            pool.close()
            pool.join()
            
            bar_queue.close()
            bar_finish.close()
            
    except Exception as e:
        print(f"Error during multiprocessing: {e}")
        
    finally:
        print('Closing resources...')
        if is_save and not is_visual:
            f.close()
        


    print('Instance counts:', '\n'+readable_dict(instance_count))

    if is_visual:
        print("Collecting data ...")
        collector.compose()
        collector.numpy()
        print("Writing data to disk...")
        collector.save(os.path.join('.', f'{mode}_sdf_{target_dataset.get_dataset_name()}_{target_dataset.category_name}_p{sample_points}_s{sigma}_a{scale_range}_l{dataset_len}_u{is_scale_unit}_onepart.pkl'))
        
    print("Done.")
