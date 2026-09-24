import os, sys
from scipy.spatial.transform import Rotation as Rt
import numpy as np
import cv2
from PIL import Image
# from hoi4d_dataset_utils import fill_missing
import trimesh
import json
from copy import deepcopy
import natsort
import torch
import tqdm
import pickle
import time
import gzip
import importlib
'''
./ZY2021080000*/H*/C*/N*/S*/s*/T*/
     |--align_rgb
         |--image.mp4
     |--align_depth
         |--depth_video.avi
     |--objpose
         |--*.json
     |--action
         |--color.json
     |--3Dseg
         |--raw_pc.pcd
         |--output.log
         |--label.pcd
     |--2Dseg
         |--mask
'''


# C3 C4 C6 C8 C9 C11 C14 C17 C18


blacklist = [
    'ZY20210800002/H2/C6/N29/S246/s03/T3',
    'ZY20210800003/H3/C3/N44/S284/s01/T2',
    'ZY20210800004/H4/C3/N63/S379/s02/T1',
    'ZY20210800001/H1/C8/N26/S317/s04/T2',
    'ZY20210800001/H1/C8/N15/S73/s03/T1'
]

mapping = [
     '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
     'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
     'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
]

category_pose_order = {
    "C1": [],
    "C2": [],
    # "C3": [['Laptopkeyboard'], ['Laptopdisplay']],
    "C3": [['Laptopdisplay'], ['Laptopkeyboard']], # reserved
    "C4": [['Lockerbody', 'Safebox'], ['Lockersldingdoor'], ['Lockerdrawer']],
    "C5": [],
    "C6": [['Safebox'], ['Safedoor']],
    "C7": [],
    "C8": [['bucket'], ['Buckethandle']],
    "C9": [],
    "C11": [],
    "C12": [],
    "C13": [],
    "C14": [['Dustbinbase'], ['Dustbincover']],
    "C17": [],
    "C18": [],
    "C20": [],
}

obj_data_order = {
    "C1": [],
    "C2": [],
    "C3": [['base_frame'], ['screen_frame']], # not reserved
    # "C3": [['screen_frame'], ['base_frame']],
    "C4": [['vertical_side_panel'], ['cabinet_door_surface'], ['drawer_front']],
    "C5": [],
    "C6": [['frame'], ['door_frame']],
    "C7": [],
    "C8": [['body'], ['handle']],
    "C9": [],
    "C11": [],
    "C12": [],
    "C13": [],
    "C14": [['body'], ['lid']],
    "C17": [],
    "C18": [],
    "C20": [],
}

label_order = {
    "C1": [],
    "C2": [],
    "C3": [0, 1], # reserved
    "C4": [1, 0, 2],
    "C5": [],
    "C6": [0, 1],
    "C7": [],
    "C8": [1, 0],
    "C9": [],
    "C11": [],
    "C12": [],
    "C13": [],
    "C14": [0, 1],
    "C17": [],
    "C18": [],
    "C20": [],
}

category2name_map = {
    "C1": "ToyCar",
    "C2": "Mug",
    "C3": "Laptop",
    "C4": "StorageFurniture",
    "C5": "Bottle",
    "C6": "Safe",
    "C7": "Bowl",
    "C8": "Bucket",
    "C9": "Scissors",
    "C11": "Pliers",
    "C12": "Kettle",
    "C13": "Knife",
    "C14": "TrashCan",
    "C17": "Lamp",
    "C18": "Stapler",
    "C20": "Chair"
}

category2label_map = {
    "C1" : [24, 47],
    "C2" : [17, 47, 17],
    "C3" : [25, 47, 25],
    "C4" : [19, 47, 19, 19, 16], 
    "C5" : [16, 47, 17],
    "C6" : [27, 47, 27, 16],
    "C7" : [15, 47, 46],
    "C8" : [9, 47, 9, 9, 9, 47],
    "C9" : [48, 47, 48],
    "C11": [23, 47, 23],
    "C12": [21, 47, 17],
    "C13": [31, 47, 46],
    "C14": [10, 47, 10, 16],
    "C17": [29, 47, 29, 29, 29, 47],
    "C18": [28, 47, 28, 28, 46],
    "C20": [2, 47, 47]
}


# The i-th number of each category represents the instance label number corresponding to the ith color (extended backward based on the original instance number)
category2label_map_instanceseg = {
    "C1" : [1, 2],
    "C2" : [1, 2, 3],
    "C3" : [1, 2, 1],
    "C4" : [1, 2, 1, 1, 3], 
    "C5" : [1, 2, 3],
    "C6" : [1, 2, 1, 3],
    "C7" : [1, 2, 3],
    "C8" : [1, 2, 1, 3, 3, 4],
    "C9" : [1, 2, 1],
    "C10" : [1, 2],
    "C11": [1, 2, 1],
    "C12": [1, 2, 3],
    "C13": [1, 2, 3],
    "C14": [1, 2, 1, 3],
    "C15": [1, 2, 1, 1, 1, 1],
    "C16": [1, 2, 1, 1],
    "C17": [1, 2, 1, 1, 1, 3],
    "C18": [1, 2, 1, 1, 3],
    "C19": [1, 2, 1, 1, 1, 1],
    "C20": [1, 2, 3]
}

category2numparts_map = {

    "C3": 2,
    "C4": 3,
    "C6": 2,
    "C8": 2,
    "C9": 2,
    "C11": 2,
    "C14": 2,
    "C17": 4,
    "C18": 2,

}

category2jointtype_map = {

    "C3": ['revolute'],
    "C4": ['revolute', 'prismatic',],
    "C6": ['revolute'],
    "C8": ['revolute'],
    "C9": ['revolute'],
    "C11": ['revolute'],
    "C14": ['revolute'],
    "C17": ['revolute', 'revolute', 'revolute'],
    "C18": ['revolute'],

}

# Only control SDF mode
manual_not_ok_ins = {
    "C3": ['N65'],
    "C4": ['N33', 'N34', 'N35', 'N36', 'N37', 'N38', 'N39', 'N40'],
    "C6": [],
    "C8": [],
    "C9": [],
    "C11": [],
    "C14": ['N01', 'N16', 'N17', 'N23', 'N26', 'N27', 'N28', 'N29', 'N33', 'N34', 'N35', 'N40', 'N44'],
    "C17": [],
    "C18": [],
}

color = np.array([
    # [.66, .66, .56, 1],
    [.68, .28, .36, 1],
    [.35, .65, .22, 1],
    [.32, .24, .65, 1],
    [0.0, .52, .52, 1],
    [0.0, 0.0, 0.0, 1],
], dtype=np.float32)

def get_colored_cloud(cloud, part_cls):
    """
    使用 NumPy 根据零件分类为点云分配颜色。

    参数:
        cloud (np.ndarray): 点云坐标。形状为 (N, 3) 或 (B, N, 3)。
        part_cls (np.ndarray): 每个点的零件类别索引。形状为 (N,) 或 (B, N)。
        color (np.ndarray): 颜色映射。形状为 (num_colors, 3)。

    返回:
        np.ndarray: 带颜色的点云。形状为 (N, 6) 或 (B, N, 6)。
        
    """
    
    # 确保 part_cls 只是索引，没有尾随的维度 1
    if part_cls.ndim == cloud.ndim and part_cls.shape[-1] == 1:
        part_cls = np.squeeze(part_cls, axis=-1)

    # 使用 part_cls 中的索引从颜色图中选择颜色
    point_colors = color[part_cls]

    # 将点坐标与其分配的颜色连接起来
    return np.concatenate([cloud, point_colors], axis=-1)

def transform_points(hrt: np.ndarray, points: np.ndarray, auto_expand=True):
    '''
    hrt: (..., 4, 4)
    points: (..., N, 3)
    
    return: (..., N, 3)
    '''
    
    batch_shape = hrt.shape[:-2]
    N = points.shape[-2]
    
    # auto expand the points batch dimension
    if hrt.shape != points.shape and len(hrt.shape) == len(points.shape) and auto_expand:
        if sum(hrt.shape) > sum(points.shape):
            # For numpy, we need to use broadcasting or tile
            # First, expand points to match batch shape
            points_expanded = np.broadcast_to(points, (*batch_shape, N, 3))
            # Since broadcast_to returns a view, we copy if needed
            if points_expanded.shape != points.shape:
                points = np.broadcast_to(points, (*batch_shape, N, 3))
        else:
            hrt = np.broadcast_to(hrt, (*batch_shape, 4, 4))
    
    # Apply rotation: R @ points
    pr = np.einsum('...ij,...kj->...ki', hrt[..., :3, :3], points)  # (..., N, 3)
    
    # Add translation
    pr = pr + hrt[..., :3, 3][..., np.newaxis, :]  # (..., N, 3)
    
    return pr

def transform_points_by_segmentation(transform:np.ndarray, points:np.ndarray, seg:np.ndarray):
    '''
    Args:
        transform: (B, P, 4, 4) transformation matrix
        points: (B, N, 3) point cloud
        seg: (B, N, 1) segmentation mask, which contains P unique part ids for each point
    Returns:
        transformed_points: (B, N, 3) transformed point cloud
    '''

    tc = np.zeros_like(points)
    for part_id in range(transform.shape[1]):
        tc = np.where(seg == part_id,
                      transform_points(transform[:, part_id], points),
                      tc)
    return tc



def get_num_parts(category):
    return category2numparts_map.get(category, None)
def get_category_name(category):
    return category2name_map.get(category, None)
def get_category_jointtype(category):
    return category2jointtype_map.get(category, None)

class HOI4D:
    
    NAME = 'hoi4d'
    
    def __init__(self, root_dir='',
                 category='C6',
                 num_points=2048, 
                 mode='train', 
                 sdf_mode=False, sdf_sample_num=2000,
                 add_noise=True, 
                 debug=False,
                 use_cache=False,
                 cache_dir='',
                 force_write_cache=False,
                 verbose=True,
                 ):
        
        self.articulated_object_list = ['C3', 'C4', 'C6', 'C8', 'C9', 'C11', 'C14', 'C17', 'C18']
        if category not in self.articulated_object_list:
            raise ValueError(f"Category {category} is not in the list of articulated objects: {self.articulated_object_list}.")
        if mode == 'test': mode = 'val'
        
        
        self.root_dir = root_dir
        
        self.mode = mode
        self.add_noise = add_noise
        self.category = category
        self.category_name = category2name_map[self.category]
        self.sdf_mode = sdf_mode
        self.sdf_sample_num = sdf_sample_num

        self.is_debug = debug
        self.cache_root_dir = cache_dir
        self.force_write_cache = force_write_cache
        self.use_cache = use_cache
        
        if self.sdf_mode:
            if self.category == 'C6':
                fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                sys.path.append(fpath)
                module = importlib.import_module(f'{self.category_name}_blend.generator')
                self.generator = module.Generator(root_dir=os.path.join(fpath, f'{self.category_name}_blend'))
        
        if self.force_write_cache:
            # print in yellow
            print(f"\033[93mWarning: Force writing cache to {self.cache_root_dir}. Existing cache will be overwritten.\033[0m")
        
        try:
            self.datalist = self.load_list_from_txt(f'valid_arti_{self.category_name}.txt', blacklist=blacklist, verbose=verbose)
        except:
            self.datalist = self.load_list_from_txt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'valid_arti_{self.category_name}.txt'), blacklist=blacklist, verbose=verbose)

        self.frame_per_sequence = 300
        
        self.sample_num = num_points
        
        self.xmap = np.array([[i for i in range(1920)] for j in range(1080)])
        self.ymap = np.array([[j for i in range(1920)] for j in range(1080)])

        
        self.depth_scale = 1000.0
        
                
                
        # testset leak
        # test_sets = self.load_list_from_txt('testset.txt')
        # self.test_split = [d for d in self.datalist if d in test_sets]
        # self.train_split = [d for d in self.datalist if d not in test_sets]
        
        
        
        all_instance = {}
        
        for video_name in self.datalist:
            instance_id_str = video_name.split('/')[3][1:]
            if instance_id_str not in all_instance:
                all_instance[instance_id_str] = 0
            all_instance[instance_id_str] += 1

                
        # train_ins = []
        self.test_split = []
        self.train_split = []
        
        # manually split
        self.all_test_ins = {
                        'C3':['031', '046', '055', '063',],
                        'C4':['051', '063', '069', '070',],
                        # 'C6':['03', '10', '20', '30',],
                        'C6':['050', '051', '052', '054',],
                        'C8':['013', '034', '044', '048'],
                        'C9':['034', '040', '048',],
                        'C11':['039', '049', '056',],
                        'C14':['042', '045', '049', '050',],
                        # 'C14':['045', '018', '049', '050',],
                        # 'C14':['046', '048', '049', '050',],
                        'C17':['019', '056', '063', '071',],
                        'C18':['038', '047',],
                        }
        
        self.test_ins = self.all_test_ins.get(self.category, None)
        
        
        for video_name in self.datalist:
            instance_id_str = video_name.split('/')[3][1:] # NXX -> 0XX
            instance_id_str = f'{int(instance_id_str):0>3d}'
            if instance_id_str in self.test_ins:
                self.test_split.append(video_name)
            else:
                # train_ins.append(instance_id_str)
                self.train_split.append(video_name)
        
        
        
        if self.mode == 'train':
            self.datalist = self.train_split
        elif self.mode in ['test', 'val']:
            self.datalist = self.test_split
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'train', 'test', or 'val'.")
     
     

        # Check if cache exists. If it does, load it.
        # obj kinematics cache was not controlled by the force_write_cache flag
        # if os.path.exists(cache_path) and not self.force_write_cache:
        try:
            assert self.use_cache, "Not using cache for articulated kinematics loading."
            assert not self.force_write_cache, "Force write cache is enabled, load from raw."
            
            kin_cache_path = os.path.join(self.cache_root_dir, 'cache', f'kinematics_{self.category}_data.pkl')
            if verbose:
                print(f"Loading articulated kinematics data from cache: {kin_cache_path}")
            with open(kin_cache_path, 'rb') as f:
                self.objs_data = pickle.load(f)
                
            self.cam_intrin = {
                'ZY20210800001': np.load(os.path.join(self.cache_root_dir, 'ZY20210800001', 'intrin.npy')),
                'ZY20210800002': np.load(os.path.join(self.cache_root_dir, 'ZY20210800002', 'intrin.npy')),
                'ZY20210800003': np.load(os.path.join(self.cache_root_dir, 'ZY20210800003', 'intrin.npy')),
                'ZY20210800004': np.load(os.path.join(self.cache_root_dir, 'ZY20210800004', 'intrin.npy')),
            }

        # If cache does not exist, process the data and save it to cache.
        except (Exception, AssertionError) as e:
            if verbose: print(f"Processing raw data due to {e}")
            
            obj_paths = os.path.join(self.root_dir, 'cad_model', 'articulated', self.category_name)
            self.objs_data = {self.category:{}}
            for ins in tqdm.tqdm(os.listdir(obj_paths), ascii=True, desc='Loading articulated kinematics', leave=False):
                try:
                    self.objs_data[self.category][ins] = self.process_articulated_kinematics(os.path.join(obj_paths, ins))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    # traceback.print_exc()
                    if verbose: print(f"Error processing articulated kinematics for {ins} in {obj_paths}. Skipping this instance.")
                    ...
                
            self.cam_intrin = {
                'ZY20210800001': np.load(os.path.join(self.root_dir, 'ZY20210800001', 'intrin.npy')),
                'ZY20210800002': np.load(os.path.join(self.root_dir, 'ZY20210800002', 'intrin.npy')),
                'ZY20210800003': np.load(os.path.join(self.root_dir, 'ZY20210800003', 'intrin.npy')),
                'ZY20210800004': np.load(os.path.join(self.root_dir, 'ZY20210800004', 'intrin.npy')),
            }
                
                
            if self.force_write_cache \
                and self.use_cache \
                and os.path.exists(self.cache_root_dir):
                    
                if verbose: print(f"Writing cache to: {self.cache_root_dir}")
                
                kin_cache_path = os.path.join(self.cache_root_dir, 'cache', f'kinematics_{self.category}_data.pkl')
                os.makedirs(os.path.dirname(kin_cache_path), exist_ok=True)
                with open(kin_cache_path, 'wb') as f:
                    pickle.dump(self.objs_data, f)
                    
                for k, v in self.cam_intrin.items():
                    cache_path = os.path.join(self.cache_root_dir, k, f'intrin.npy')
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    np.save(cache_path, v)
                
                
        self.all_ins = list(self.objs_data[self.category].keys())
        
        not_ok_ins = manual_not_ok_ins.get(self.category, [])
        not_ok_ins = ['0'+ins_id[1:] for ins_id in not_ok_ins]
        
        self._all_ins = set(self.all_ins) - set(not_ok_ins)
        self._test_ins = set(self.test_ins)
        self._train_ins = self._all_ins - self._test_ins
        self.train_ins = list(self._train_ins)
        self.all_ins = list(self._all_ins)
        
        if verbose:
            print('-'*50)
            print(f'HOI4D Dataset subset {self.category}-{get_category_name(self.category)} in {self.mode} mode')
            print(f"    Train sequences: {len(self.train_split)}, Test sequences: {len(self.test_split)}")
            print(f"    Total instance num: {len(self.all_ins)}")
            print(f"    Train instance num: {len(self.train_ins)}")
            print(f"    Test instance num: {len(self.test_ins)}")
            # print(f"    Train instances: {self.train_ins}")
            # print(f"    Test instances: {self.test_ins}")
            print( '    Category num parts: ', get_num_parts(self.category))
            print( '    Category joint type: ', get_category_jointtype(self.category))
            print('-'*50)


    @staticmethod
    def get_dataset_name(upper=False):
        return HOI4D.NAME.upper() if upper else HOI4D.NAME

    def __len__(self):
        
        if self.sdf_mode:
            return self.sdf_sample_num
        else:
            return len(self.datalist) * self.frame_per_sequence
    
    def __getitem__(self, idx):
        
        try:
        
            if isinstance(idx, slice):
                start, stop, step = idx.indices(len(self))
                return [self[i] for i in range(start, stop, step)]
            if isinstance(idx, (list, tuple, np.ndarray)):
                return [self[i] for i in idx]
            
            if isinstance(idx, (int, np.integer)):
                if idx < 0:
                    idx = len(self) + idx
                    
            if self.sdf_mode:
                return self.get_frame_sdf(idx)
            else:
                return self.get_frame(idx)
            
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        # except IndexError:
        #     ...
            
        # except Exception as e:
        #     seq_id = idx // self.frame_per_sequence
        #     frame_id = idx % self.frame_per_sequence          
        #     video_name = self.datalist[seq_id]
        #     print(f"Error getting frame at {video_name}-{frame_id}: {e}")
        #     raise e
        
        
    '''
    NOTE:
    For C6 Safe 49 instances:
        instance 01-016 total 13 instances are likely the same one.
        instance 17-050 total 16 instances are likely the same one.
        remaining 20 instances are unique.
    '''
        
    def get_frame_sdf(self, idx):

        if idx >= self.sdf_sample_num:
            raise IndexError('Index out of range')

        # video_name = np.random.choice(self.datalist, 1)[0]
        # instance_id = int(video_name.split('/')[3][1:])
        # instance_id_str = f'{instance_id:0>3d}'
        # obj_data = self.objs_data[self.category][instance_id_str]
        
        if self.mode == 'train':
            instance_id = int(np.random.choice(self.train_ins, 1)[0])
        else:
            instance_id = int(np.random.choice(self.test_ins, 1)[0])
        instance_id_str = f'{int(instance_id):0>3d}'
        
        
        if self.mode == 'train':
            assert instance_id_str not in self.test_ins, 'oops, test instance in train mode'
        
        
        if self.category == 'C6':
            if instance_id < 50 and self.mode == 'train':
                isAug = np.random.uniform(0, 1.0, 1)[0] > 0.5
                if isAug:
                    instance_id = 0
                    instance_id_str = f'{instance_id:0>3d}'
                    obj_data = self.generator.generate(np.random.uniform(0, 1.0, self.generator.morph_num))
                else:
                    obj_data = self.objs_data[self.category][instance_id_str]
            else:
                obj_data = self.objs_data[self.category][instance_id_str]
        else:
            obj_data = self.objs_data[self.category][instance_id_str]
        
        bbox_extents = []
        bbox_transforms = []
        for mesh in obj_data['preload_meshs']:
            bbox_extents.append(np.array(mesh.bounding_box.primitive.extents, dtype=np.float32))
            bbox_transforms.append(np.array(mesh.bounding_box.primitive.transform, dtype=np.float32))
        bbox_extents = np.stack(bbox_extents, axis=0)
        bbox_transforms = np.stack(bbox_transforms, axis=0)
        
        mesh = obj_data.get('one_mesh', None)
        if mesh is None:
            raise ValueError(f"No preloaded mesh found for instance {instance_id_str} in category {self.category}.")
            # mesh = [trimesh.load_mesh(obj_path, process=False) for obj_path in obj_data['obj_paths']]
            # mesh = trimesh.util.concatenate(mesh)
            # mesh.merge_vertices()
        
        cloud = np.array(mesh.sample(self.sample_num))
        
        if self.add_noise and self.mode == 'train':
            cloud += np.clip(0.001 * np.random.randn(*cloud.shape), -0.005, 0.005)
            
        scale_factor = 1. / cloud.max() * 0.7
        
        zipped_mesh = [{
                        'vertices': np.array(mesh.vertices),
                        'faces': np.array(mesh.faces),
                        }]
        
        norm_part_loc = obj_data['joint_pivots_matrix']
        norm_part_axis = obj_data['joint_axis_matrix']
        
        
        # if self.sdf_scale_normal:
        #     norm_part_loc = norm_part_loc * scale_factor
        #     # norm_part_axis = norm_part_axis * scale_factor
        #     cloud = cloud * scale_factor
        #     zipped_mesh[0]['vertices'] *= scale_factor
        
        
        return cloud.astype(np.float32), \
                zipped_mesh, \
                np.ones((cloud.shape[0],), dtype=np.float32), \
                norm_part_loc.astype(np.float32)[1:], \
                -norm_part_axis.astype(np.float32)[1:], \
                bbox_extents.astype(np.float32), \
                bbox_transforms.astype(np.float32), \
                scale_factor.astype(np.float32), \
                np.array(instance_id, dtype=np.int32),


    def get_frame_with_name(self, video_name, frame_id):
        """
        Get a specific frame from a video sequence by its name and frame ID.
        
        Args:
            video_name (str): The name of the video sequence.
            frame_id (int): The frame ID to retrieve.
        
        Returns:
            dict: A dictionary containing the RGB image, depth image, masks, pose, and point cloud.
        """
        
        idx = self.datalist.index(video_name) * self.frame_per_sequence + frame_id
        return self.get_frame(idx)

        
    def get_frame(self, idx):
                
        seq_id = idx // self.frame_per_sequence
        frame_id = idx % self.frame_per_sequence
                        
        video_name = self.datalist[seq_id]
        camera_name = video_name.split('/')[0]
        
        cache_not_found = False
        
        if self.use_cache and not self.force_write_cache:
            try:
                cache_dir = os.path.join(self.cache_root_dir, video_name)
                cache_path = os.path.join(cache_dir, f'{frame_id:0>5d}.pkl.gz')
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        data = pickle.loads(gzip.decompress(f.read()))
                        
                    rgb = data['rgb']
                    crop_label_mask = data['label_mask']
                    pts = data['pts']
                    choose = data['choose']
                    rt_per_part = data['rt_per_part']
                    size = data['size']
                
                    inlier_ind = np.arange(pts.shape[0])
                    
                else:
                    cache_not_found = True
            except:
                cache_not_found = True
    
            
        if not self.use_cache or cache_not_found or self.force_write_cache:
            rgb_folder = os.path.join(self.root_dir, video_name, 'align_rgb')
            depth_folder = os.path.join(self.root_dir, video_name, 'align_depth')
            objpose_folder = os.path.join(self.root_dir, video_name, 'objpose')
            seg2D_folder = os.path.join(self.root_dir, video_name, '2Dseg', 'mask' if camera_name=='ZY20210800001' else 'shift_mask')
            seg2D_path = os.path.join(seg2D_folder, f'{frame_id:0>5d}.png')


            
            rgb = cv2.imread(os.path.join(rgb_folder, f'{frame_id:0>5d}.jpg'))[:, :, ::-1]
            depth = cv2.imread(os.path.join(depth_folder, f'{frame_id:0>5d}.png'), cv2.IMREAD_UNCHANGED)
            raw_mask = cv2.imread(seg2D_path)[:, :, ::-1]
            masks = self.get_mask_and_label_only_arti(raw_mask, self.category, camera=camera_name)
            label_mask = self.combine_binary_masks_to_labeled(masks)
            union_mask = np.zeros_like(masks[0])
            for mask in masks:
                union_mask = np.logical_or(union_mask, mask)
            isl = self.find_mask_islands(union_mask, min_area=1)
            if len(isl) > 1:
                self.remove_islands(isl, union_mask)

            try:
                posefile_path = os.path.join(objpose_folder, f'{frame_id:d}.json')
                rot, trans, size = self.read_rtd_arti(posefile_path)
            except:
                posefile_path = os.path.join(objpose_folder, f'{frame_id:0>5d}.json')
                rot, trans, size = self.read_rtd_arti(posefile_path)
            rt_per_part = self.compose_rt(rot, trans)
                    
            
            # if self.force_write_cache:
                
            #     cache_rgb_folder = os.path.join(self.cache_root_dir, video_name, 'align_rgb')
            #     cache_depth_folder = os.path.join(self.cache_root_dir, video_name, 'align_depth')
            #     cache_objpose_folder = os.path.join(self.cache_root_dir, video_name, 'objpose')
            #     cache_seg2D_folder = os.path.join(self.cache_root_dir, video_name, '2Dseg', 'mask' if camera_name=='ZY20210800001' else 'shift_mask')
                
            #     union_mask_path = os.path.join(self.cache_root_dir, video_name, 'union_mask', f'{frame_id:0>5d}.png')
            #     label_mask_path = os.path.join(self.cache_root_dir, video_name, 'label_mask', f'{frame_id:0>5d}.png')

            #     os.makedirs(cache_rgb_folder, exist_ok=True)
            #     os.makedirs(cache_depth_folder, exist_ok=True)
            #     os.makedirs(cache_objpose_folder, exist_ok=True)
            #     os.makedirs(cache_seg2D_folder, exist_ok=True)
            #     os.makedirs(os.path.dirname(label_mask_path), exist_ok=True)
            #     os.makedirs(os.path.dirname(union_mask_path), exist_ok=True)
                
            #     cmd = f'''
            #     cp {os.path.join(rgb_folder, f"{frame_id:0>5d}.jpg")} {cache_rgb_folder};
            #     cp {os.path.join(depth_folder, f"{frame_id:0>5d}.png")} {cache_depth_folder};
            #     cp {posefile_path} {cache_objpose_folder};
            #     cp {seg2D_path} {cache_seg2D_folder};
            #     '''
            #     # # copy files to cache directory
            #     os.system(cmd)
                
            #     cv2.imwrite(union_mask_path, (union_mask * 255).astype(np.uint8))
            #     cv2.imwrite(label_mask_path, label_mask.astype(np.uint8))

            # depth = fill_missing(depth, self.depth_scale, 1)
                            
            

            _mask_pixel_id = np.where(union_mask)
            box = [np.min(_mask_pixel_id[1]), np.min(_mask_pixel_id[0]), np.max(_mask_pixel_id[1]), np.max(_mask_pixel_id[0])]
            
            cmin, cmax, rmin, rmax = self.get_bbox(box)
            union_mask = np.logical_and(union_mask , depth > 0)
            choose = union_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            
            cam_int = self.cam_intrin[camera_name]
            cam_fx = cam_int[0, 0]
            cam_fy = cam_int[1, 1]
            cam_cx = cam_int[0, 2]
            cam_cy = cam_int[1, 2]
            
            
            
            # create point cloud by all points in the mask
            pts_z = depth.copy()[rmin:rmax, cmin:cmax].reshape((-1))[choose] / self.depth_scale
            pts_x = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cx) * pts_z / cam_fx
            pts_y = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cy) * pts_z / cam_fy
            pts = np.transpose(np.stack([pts_x, pts_y, pts_z]), (1,0)).astype(np.float32) # 480*640*3
            
            # remove outliers
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts)
            # inlier_cloud, inlier_ind = pcd.remove_statistical_outlier(
            #     nb_neighbors=50, 
            #     std_ratio=2.0
            # )
            inlier_ind = np.arange(pts.shape[0])
            crop_label_mask = label_mask[rmin:rmax, cmin:cmax]
            choose = choose[inlier_ind]
        
            if self.force_write_cache or (self.use_cache and cache_not_found):
                data = {
                    'video_name': video_name,
                    'frame_id': frame_id,
                    'pts': pts,
                    'choose': choose,
                    'rgb': rgb[rmin:rmax, cmin:cmax],
                    'label_mask': crop_label_mask,
                    'rt_per_part': rt_per_part,
                    'size': size
                }
                cache_dir = os.path.join(self.cache_root_dir, video_name)
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, f'{frame_id:0>5d}.pkl.gz'), 'wb') as f:
                    f.write(gzip.compress(pickle.dumps(data), compresslevel=4))        
                    
        
        if len(choose)<=0:
            print('choose is empty')
            print('video_name: ', video_name)
            print('frame_id: ', frame_id)
            return None
        
        elif len(choose) <= self.sample_num:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
        choose = choose[choose_idx]
        inlier_ind = np.array(inlier_ind, dtype=np.int32)[choose_idx]
        pts = pts[inlier_ind]
        
        choose_mask = crop_label_mask.reshape((-1))[choose]
        choose_mask -= 1 # make sure the mask starts from 0
        # choose_rgb = rgb[rmin:rmax, cmin:cmax].reshape((-1, 3))[choose] / 255.0

        if self.add_noise:
            pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)



        instance_id = int(video_name.split('/')[3][1:])
        instance_id_str = f'{instance_id:0>3d}'
        obj_data = self.objs_data[self.category][instance_id_str]
        
        norm_part_loc = obj_data['joint_pivots_matrix']
        norm_part_axis = obj_data['joint_axis_matrix']
        
        align_t = np.stack(obj_data['obj_align_t'], axis=0)
        rt_per_part = rt_per_part @ align_t # for not align obj
        

        # rgb = cv2.rectangle(rgb.copy(), (rmin, cmin), (rmax, cmax), (0, 255, 0), 2)

        # pts_color = get_colored_cloud(pts, choose_mask)
        
        joint_status = []
        refined_pose = []
        connection_matrix = obj_data['connection_matrix']
        
        for part_id in range(connection_matrix.shape[0]):
            
            parent_index = np.where(connection_matrix[part_id] == 1)[0]
            jtype = obj_data['joint_types'][part_id]
            
            if len(parent_index) > 0:
                parent_index = parent_index[0]
                status = self.get_status_from_part_pose(parent_pose=rt_per_part[parent_index], 
                                                        part_pose=rt_per_part[part_id], 
                                                        joint_pivot=norm_part_loc[part_id], 
                                                        joint_axis=norm_part_axis[part_id], 
                                                        joint_type=jtype)
                
                
                joint_status.append(status)
                refine_part_pose = self.get_part_pose_from_status(parent_pose=rt_per_part[parent_index], 
                                                                  joint_pivot=norm_part_loc[part_id], 
                                                                  joint_axis=norm_part_axis[part_id], 
                                                                  joint_status=status, 
                                                                  joint_type=jtype)
                refined_pose.append(refine_part_pose)
            else:
                joint_status.append(0)
                refined_pose.append(rt_per_part[part_id])
                
        
        refined_pose = np.stack(refined_pose, axis=0)
        joint_status = np.array(joint_status, dtype=np.float32)
        joint_limits = obj_data['joint_limits']

        complete_cloud = obj_data['complete_cloud']
        complete_cloud_seg = obj_data['complete_cloud_seg']
        complete_cloud_per_part = obj_data['complete_cloud_per_part']
        
        if self.is_debug:
            
            if self.use_cache:
                raise ValueError("Debug mode is not compatible with cache mode.")
            
            pts_z = depth.copy().reshape((-1)) / self.depth_scale
            pts_x = (self.xmap.reshape((-1)) - cam_cx) * pts_z / cam_fx
            pts_y = (self.ymap.reshape((-1)) - cam_cy) * pts_z / cam_fy
            raw_pts = np.transpose(np.stack([pts_x, pts_y, pts_z]), (1,0)).astype(np.float32) # 480*640*3

            # canoni_mesh = [trimesh.load_mesh(obj_path, process=False) for obj_path in obj_data['obj_paths']]
            canoni_mesh = obj_data['preload_meshs']
            for i in range(len(canoni_mesh)):
                canoni_mesh[i].visual.vertex_colors = np.ones_like(canoni_mesh[i].visual.vertex_colors) * color[i]
            observe_mesh = [deepcopy(mesh).apply_transform(rt) for mesh, rt in zip(canoni_mesh, refined_pose)]

            rgb_patch = rgb[cmin:cmax, rmin:rmax]
            
            colored_raw_pts = np.concatenate((raw_pts, rgb.reshape((-1, 3))/255.), axis=1)[::5]

            data = {
                'pts':pts,
                'choose_mask':choose_mask,
                'refined_pose':refined_pose,
                'joint_status':joint_status,
                'norm_part_loc':norm_part_loc[1:],
                'norm_part_axis':-norm_part_axis[1:],
                'joint_limits':joint_limits,
                'rgb_patch':rgb_patch,
                'complete_cloud':complete_cloud,
                'complete_cloud_seg':complete_cloud_seg,
                'complete_cloud_per_part':complete_cloud_per_part,
                'rgb':rgb,
                'depth':depth,
                'mask':union_mask,

                # 'raw_mask':raw_mask,
                # 'rt':rt_per_part,
                
                'raw_pts':colored_raw_pts,
                # 'pts':get_colored_cloud(pts, choose_mask),
                'joint_pivots_matrix':obj_data['joint_pivots_matrix'],
                'joint_axes_line':np.stack([obj_data['joint_pivots_matrix'], 
                                                obj_data['joint_pivots_matrix'] + obj_data['joint_axis_matrix']], axis=1),

                'video_name':video_name,
                'frame_id':frame_id,
                'cam_int':cam_int,
                # 'complete_cloud':get_colored_cloud(complete_cloud, complete_cloud_seg),
                # 'complete_cloud_per_part':complete_cloud_per_part,
                # 'project_cloud': get_colored_cloud(transform_points_by_segmentation(np.linalg.inv(refined_pose)[None], pts[None], choose_mask[None, :, None])[0],
                                                #    choose_mask),

            }
            for i, mesh in enumerate(observe_mesh):
                data.update({f'observe_mesh_{i}': mesh})
            for i, mesh in enumerate(canoni_mesh):
                data.update({f'canonical_mesh_{i}': mesh}) 
                
            data.update({
                'one_mesh': obj_data.get('one_mesh', None),
            })       
            # print('processed')
            return data
        
        else:


            # if pts is None:
            #     print('pts is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if choose_mask is None:
            #     print('choose_mask is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if refined_pose is None:
            #     print('refined_pose is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if joint_status is None:
            #     print('joint_status is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if norm_part_loc is None:
            #     print('norm_part_loc is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if norm_part_axis is None:
            #     print('norm_part_axis is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if joint_limits is None:
            #     print('joint_limits is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            # if complete_cloud is None:
            #     print('complete_cloud is None')
            #     print('video_name: ', video_name)
            #     print('frame_id: ', frame_id)
            
            
            
            
            
            return  torch.tensor(pts, dtype=torch.float32), \
                    torch.tensor(choose_mask, dtype=torch.int64), \
                    torch.tensor(refined_pose, dtype=torch.float32), \
                    torch.tensor(joint_status, dtype=torch.float32), \
                    torch.tensor(norm_part_loc[1:], dtype=torch.float32), \
                    torch.tensor( - norm_part_axis[1:], dtype=torch.float32), \
                    torch.tensor(joint_limits, dtype=torch.float32), \
                    torch.tensor(complete_cloud, dtype=torch.float32), \
                    torch.tensor(complete_cloud_seg, dtype=torch.int64), \
                    torch.tensor(complete_cloud_per_part, dtype=torch.float32), \
                    video_name, \
                    str(frame_id), \
                    torch.tensor(connection_matrix, dtype=torch.float32), \




    def _check_dataset(self, ):
        raise NotImplementedError
        train_datalist_path = 'all.txt'
        test_datalist_path = 'testset.txt'
        
        train_datalist = self.load_list_from_txt(train_datalist_path)
        print('train seqlen:', len(train_datalist))
        
        valid_train_datalist = self.check_is_valid(self.root_dir, train_datalist)
        
        with open('valid_train_all.txt', "w", encoding='UTF-8') as f:
            for d in valid_train_datalist:
                f.write(d + "\n")
                
        valid_train_datalist_rigid = self.select_rigid_seq(valid_train_datalist, self.rigid_object_list)
        with open('valid_train_rigid.txt', "w", encoding='UTF-8') as f:
            for d in valid_train_datalist_rigid:
                f.write(d + "\n")

                
        print('vaild:', len(valid_train_datalist))
        print('train seqlen:', len(train_datalist))
        
        
        test_datalist = self.load_list_from_txt(test_datalist_path)
        print('test seqlen:', len(test_datalist))
        
        valid_test_datalist = self.check_is_valid(self.root_dir, test_datalist)
        
        with open('valid_test_all.txt', "w", encoding='UTF-8') as f:
            for d in valid_test_datalist:
                f.write(d + "\n")
                
        valid_test_datalist_rigid = self.select_rigid_seq(valid_test_datalist, self.rigid_object_list)
        with open('valid_test_rigid.txt', "w", encoding='UTF-8') as f:
            for d in valid_test_datalist_rigid:
                f.write(d + "\n")

                
        print('vaild:', len(valid_test_datalist))
        print('test seqlen:', len(test_datalist))


    @staticmethod
    def shift_mask(mask, ZY):
        if ZY == "ZY20210800001":
            dx, dy = 0, 0
        elif (ZY == "ZY20210800002") or (ZY == "ZY20210800004"):
            dx, dy = 15, -3
        elif ZY == "ZY20210800003":
            dx, dy = 15, -30
        else:
            raise NotImplementedError

        rows, cols, _ = mask.shape
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])
        dst = cv2.warpAffine(mask, MAT, (cols, rows))
        return dst

        
    @staticmethod
    def load_list_from_txt(path, blacklist=None, verbose=True):
        datalist = []
        if blacklist is None:
            blacklist = []
        blacklist_cout = 0
        with open(path, "r") as f:
            for line in f:
                    content = line.strip()
                    if len(content) == 0:
                        continue
                    if content in blacklist:
                        blacklist_cout += 1
                        continue
                    datalist.append(content)
        if verbose: print(f"remove seqs from blacklist: {blacklist_cout}")
        return datalist
        
    @staticmethod    
    def check_is_valid(root_dir:str, datalist:str):
        cnt = 0
        valid_datalist = []
        for video_name in datalist:
            has_file = 0
            for i in range(320):
                    if os.path.isfile(os.path.join(root_dir, video_name, 'align_rgb', f'{i:0>5d}' + '.jpg')) and \
                        os.path.isfile(os.path.join(root_dir, video_name, 'align_depth', f'{i:0>5d}' + '.png')):
                        has_file += 1
            if has_file == 300:
                cnt += 1
                valid_datalist.append(video_name)
            else:
                print(video_name, 'len', has_file)
        return sorted(valid_datalist)
                

    def get_mask_and_label_only_arti(self, image, category, camera):
        
    
        
        assert image.shape == (1080, 1920, 3)
        # image = self.shift_mask(image, camera)
        color_map = self.get_color_map()

        arrs = []
        categories = []  # semantic segmentation
        labels_instanceseg = []  # instance segmentation
        
        unique_id = category2label_map[category][0]
        instance_id = category2label_map_instanceseg[category][0]

        for i in range(len(category2label_map[category])):
            
            # skip hand mask
            if category2label_map[category][i] != unique_id: # remove other mask
                continue
            if category2label_map_instanceseg[category][i] != instance_id: # remove other instance
                continue
            
            color = color_map[i + 1]
            valid = (image[..., 0] == color[0]) & (image[..., 1] == color[1]) & (image[..., 2] == color[2])
            # if np.sum(valid) == 0:
            #     continue
            arrs.append(valid)
            # categories.append(category2label_map[category][i])
            # labels_instanceseg.append(category2label_map_instanceseg[category][i])
            
        if len(label_order[category]):
            assert len(arrs) == len(label_order[category]), 'label_order miss match'
            
            arrs = [arrs[i] for i in label_order[category]]
            # categories = [categories[i] for i in label_order[category]]
            # labels_instanceseg = [labels_instanceseg[i] for i in label_order[category]]
        
        return arrs  #, np.array(categories), np.array(labels_instanceseg), image
        

    @staticmethod
    def combine_binary_masks_to_labeled(masks):


        labeled_mask = np.zeros_like(masks[0], dtype=np.int32)

        for i, mask in enumerate(masks):
            label = i + 1
            labeled_mask[mask == 1] = label

        return labeled_mask
    



    @staticmethod
    def get_color_map(N=256):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap

    @staticmethod
    def select_rigid_seq(pathlist, rigid_object_list):
        rigid_list = []
        for path in pathlist:
            l = path.split('/')
            if l[2] in rigid_object_list:
                rigid_list.append(path)
                
        return rigid_list



    
    def read_rtd_arti(self, file,):
        with open(file, 'r') as f:
            cont = f.read()
            cont = eval(cont)
        if "dataList" in cont:
            anno = cont["dataList"]
        else:
            anno = cont["objects"]

        trans_list = []
        rot_list  = []
        size_list  = []

        if len(category_pose_order[self.category]):
            for part_name in category_pose_order[self.category]:
                part_anno = next((item for item in anno if item['label'] in part_name), None)
                if part_anno is None:
                    raise ValueError(f"Part '{part_name}' not found in annotations.\n raw data: {anno} \n file: {file}")
                
                trans, rot, dim = part_anno["center"], part_anno["rotation"], part_anno["dimensions"]
                trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
                rot = np.array([rot['x'], rot['y'], rot['z']])
                size = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
                rot = Rt.from_euler('XYZ', rot).as_matrix()
                
                rot_list.append(np.array(rot, dtype=np.float32))
                trans_list.append(trans)
                size_list.append(size)

        else:
            for part_anno in anno:

                trans, rot, dim = part_anno["center"], part_anno["rotation"], part_anno["dimensions"]
                trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
                rot = np.array([rot['x'], rot['y'], rot['z']])
                size = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
                rot = Rt.from_euler('XYZ', rot).as_matrix()
                
                rot_list.append(np.array(rot, dtype=np.float32))
                trans_list.append(trans)
                size_list.append(size)
            
        rot = np.stack(rot_list, axis=0)
        trans = np.stack(trans_list, axis=0)
        size = np.stack(size_list, axis=0)

        return rot, trans, size

    @staticmethod
    def get_bbox(bbox):
        """ Compute square image crop window. """
        y1, x1, y2, x2 = bbox
        img_width = 1920
        img_length = 1080
        window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
        window_size = min(window_size, 880)
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        rmin = center[0] - int(window_size / 2)
        rmax = center[0] + int(window_size / 2)
        cmin = center[1] - int(window_size / 2)
        cmax = center[1] + int(window_size / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_width:
            delt = rmax - img_width
            rmax = img_width
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax

 
    def process_articulated_kinematics(self, instance_folder='', preload=True):
        """
        
        load articulated kinematics data from the given instance folder.
        
        Args:
            instance_folder (str): instance folder path, containing 'mobility_v2.json' and 'result.json'.
            preload (bool): if True, preload the data into memory.
        Returns:
            dict: dictionary containing part information, including:
                - 'part_ids': list of part IDs.
                - 'part_names': list of part names.
                - 'joint_types': list of joint types.
                - 'joint_pivots_matrix': numpy array of joint pivot points.
                - 'joint_axis_matrix': numpy array of joint axes.
                - 'connection_matrix': numpy array of connection matrix.
                - 'obj_paths': list of object paths.
                - 'obj_align_t': list of object alignment transformations.
        Raises:
            ValueError: if the joint type is unsupported.
        """
        # 1. Load JSON files
        def find_obj_paths(node, path_map):
            """Recursively traverse the result.json structure to find obj paths."""
            if "objs" in node and node["objs"]:
                # We assume the 'name' at this level is the part name we can map
                path_map[node['name']] = [f"{obj}.obj" for obj in node['objs']]
                # path_map[node['name']] = [f"{obj}-align.obj" for obj in node['objs']]
            if "children" in node:
                for child in node["children"]:
                    find_obj_paths(child, path_map)
                    
                    
        with open(os.path.join(instance_folder, 'mobility_v2.json'), 'r', encoding='utf-8') as f:
            mobility_data = json.load(f)
        with open(os.path.join(instance_folder, 'result.json'), 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        # 2. Extract OBJ paths from result.json
        obj_path_map = {}
        for root in result_data:
            find_obj_paths(root, obj_path_map)

        # 3. Process mobility_v2.json to get connection and joint data
        parts_data = {}
        for part_info in mobility_data:
            # part_id = part_info['id']
            part_name = part_info['name']
            # part_id = obj_data_order[self.category][part_name]
            for pid, name in enumerate(obj_data_order[self.category]):
                if part_name in name:
                    part_id = pid
                    break

            parts_data[part_id] = {
                "ori_id": part_info['id'],
                "name": part_name,
                "jtype": part_info.get('joint', None),
                "parent": part_info['parent'],
                "joint_params": part_info.get('jointData', {}),
                "obj_path": obj_path_map.get(part_name, ["N/A"])[0] # Get the first obj path
            }

        # 4. Generate the required output in numpy format

        parts_data = dict(natsort.natsorted(parts_data.items(), key=lambda x: x[0]))
        
        part_ids = list(parts_data.keys())
        # id_to_index = {part_id: i for i, part_id in enumerate(part_ids)}
        num_parts = len(part_ids)

        ori_id_to_index = {data['ori_id']: i for i, data in parts_data.items()}

        # Connection matrix (NxN)
        connection_matrix = np.zeros((num_parts, num_parts), dtype=int)
        # for part_id, data in parts_data.items():
        #     parent_id = data['parent']
        #     ori_id = data['ori_id']
        #     if parent_id in id_to_index:
        #         part_idx = id_to_index[part_id]
        #         parent_idx = id_to_index[parent_id]
        #         connection_matrix[part_idx, parent_idx] = 1
        #         connection_matrix[parent_idx, part_idx] = 1 
        for part_id, data in parts_data.items():
            parent_id = data['parent']
            parent_index = ori_id_to_index.get(parent_id, -1)
            if parent_index in parts_data:
                
                
                # connection_matrix[parent_index][part_id] = 1
                connection_matrix[part_id][parent_index] = 1 
            

        jtype_map = {
            '静态':'free',
            '沉重':'free',
            '自由':'free',
            '铰链（旋转）':'revolute',
            '滑动':'prismatic',
        }

        # Adjacency list (simpler representation)
        # adjacency_list = {part_id: data['parent'] for part_id, data in parts_data.items()}

        # Joint parameters ((N-1)x3)
        # Sort non-root parts with joints by ID
        # joint_part_ids = sorted([pid for pid in part_ids if parts_data[pid]['parent'] != -1])
        joint_part_ids = part_ids
        joint_pivots = []
        joint_axis = []
        joint_types = []
        joint_min = []
        joint_max = []
        for part_id in joint_part_ids:
            part_data = parts_data[part_id]
            joint_params = part_data['joint_params']
            
            jtype = jtype_map.get(part_data['jtype'], None)
            if jtype is None:
                print(f"Unsupported joint type: {part_data['jtype']}")
                raise ValueError(f"Unsupported joint type: {part_data['jtype']}")
            joint_types.append(jtype)
            
            
            if 'axis' in joint_params and 'origin' in joint_params['axis'] and 'direction' in joint_params['axis']:
                joint_pivots.append(joint_params['axis']['origin'])
                _axis = np.array(joint_params['axis']['direction'])
                joint_axis.append(_axis / np.linalg.norm(_axis))
                
                _jmin = joint_params['limit'].get('b', [0.0])
                _jmax = joint_params['limit'].get('a', [0.0])
                
                if _jmin > _jmax:
                    # Swap to ensure min <= max
                    _jmin, _jmax = _jmax, _jmin
                    # print(f"Warning: Swapped joint limits for {instance_folder} to ensure min <= max.")
                
                joint_min.append(_jmin)
                joint_max.append(_jmax)
                
            else:
                ...
                # If non-root parts do not have joint data, append zeros.
                joint_pivots.append([0.0, 0.0, 0.0])
                joint_axis.append(np.array([0.0, 0.0, 0.0]))
                joint_min.append(0.0)
                joint_max.append(0.0)

        joint_pivots_matrix = np.array(joint_pivots)
        joint_axis_matrix = np.array(joint_axis)
        joint_min = np.array(joint_min)
        joint_max = np.array(joint_max)
        
        obj_paths_list = [parts_data[pid]['obj_path'] for pid in part_ids]
        joint_limits = np.stack([joint_min, joint_max], axis=1)
        
        obj_paths = [os.path.join(instance_folder, 'objs', i) for i in obj_paths_list]

        obj_align_t_data = [np.load(os.path.join(instance_folder, 'objs', f"{os.path.splitext(i)[0]}-align-t.npy")) for i in obj_paths_list]
        try:
            obj_align_t2_data = [np.load(os.path.join(instance_folder, 'objs', f"{os.path.splitext(i)[0]}-align-t2.npy")) for i in obj_paths_list]
            for i in range(1, len(obj_align_t2_data)):
                
                # NOTE: Part transforms does not multiply the root part's transform, so we need to transform every part by the root part's transform here.
                # We assume the first part is the root part.
                obj_align_t2_data[i] = obj_align_t2_data[i] @ obj_align_t2_data[0]
            
            
        except:
            print("No align-t2 file, using identity matrix instead.")
            obj_align_t2_data = [np.eye(4, dtype=np.float32) for _ in obj_paths_list]
        finally:
            obj_align_t_data = [align_t1 @ align_t2 for align_t1, align_t2 in zip(obj_align_t_data, obj_align_t2_data)]
            # ...

        # Also, we need to transform the joint pivots and axes by the root part's align_t2
        inv_root_align_t2 = np.linalg.inv(obj_align_t2_data[0])

        for i in range(joint_pivots_matrix.shape[0]):
            jtype = joint_types[i]
            
            # only revolute joint need to be transformed
            if jtype == 'revolute':
                joint_pivots_matrix[i] = inv_root_align_t2[:3, :3] @ joint_pivots_matrix[i] + inv_root_align_t2[:3, 3]

            # temporarily, we do not transform joint axis


        if 'Laptop' in instance_folder:
            # Since laptop base is usually unseen, we reverse the order to make screen as part 0
            ...
            obj_paths = obj_paths[::-1]
            obj_align_t_data = obj_align_t_data[::-1]
            obj_align_t2_data = obj_align_t2_data[::-1]
            joint_limits = -joint_limits[:, ::-1]


        # 5. Return the processed data
        data = {
            "connection_matrix": connection_matrix,
            "joint_pivots_matrix": joint_pivots_matrix,
            "joint_axis_matrix": joint_axis_matrix,
            "joint_limits": joint_limits,
            "joint_types": joint_types,
            "obj_paths": obj_paths,
            "obj_align_t": obj_align_t_data,
        }
        

        
        if preload:
            preload_meshs = []
            for path, corr_mat in zip(obj_paths, obj_align_t2_data):
                mesh = trimesh.load_mesh(path, process=False)
                mesh.apply_transform(np.linalg.inv(corr_mat))
                preload_meshs.append(mesh)
        
            
        
            one_mesh = trimesh.util.concatenate(preload_meshs)
            one_mesh.merge_vertices()
            
            data['preload_meshs'] = preload_meshs
            data['one_mesh'] = one_mesh
            
            # complete_cloud = one_mesh.sample(self.sample_num)
            # data['complete_cloud'] = np.array(complete_cloud, dtype=np.float32)
            
            # 1. 对preload_meshs中的每个mesh都采样self.sample_num个点，并记录label
            all_points = []
            all_labels = []
            for i, mesh in enumerate(preload_meshs):
                sampled_points = mesh.sample(self.sample_num)  # 采样点
                labels = np.full((sampled_points.shape[0],), i, dtype=np.int32)  # 标签，第i个mesh标签为i
                all_points.append(sampled_points)
                all_labels.append(labels)
            
            # 2. cat所有采样点
            all_points_concat = np.concatenate(all_points, axis=0)  # [total_points, 3]
            all_labels_concat = np.concatenate(all_labels, axis=0)   # [total_points,]
            
            # 3. 使用fastest point sample采样self.sample_num个点，同时使用索引提取点label
            # 转换为torch tensor以使用fps函数
            # points_tensor = torch.from_numpy(all_points_concat).unsqueeze(0).float()  # [1, total_points, 3]
            # fps_points, fps_idx = self.fps(points_tensor, self.sample_num, needidx=True)
            fps_points_np, fps_idx_np = self.fps(all_points_concat, self.sample_num)
            
            # 提取对应的labels
            # fps_idx_np = fps_idx.squeeze(0).cpu().numpy()  # [self.sample_num,]
            fps_labels = all_labels_concat[fps_idx_np]      # [self.sample_num,]
            
            # 转换回numpy
            # fps_points_np = fps_points.squeeze(0).cpu().numpy()  # [self.sample_num, 3]
            
            data['complete_cloud'] = fps_points_np.astype(np.float32)
            data['complete_cloud_seg'] = fps_labels.astype(np.int32)
            data['complete_cloud_per_part'] = np.stack(all_points, axis=0).astype(np.float32)
            
        return data
        
        
    @staticmethod
    def compose_rt(rotation:np.ndarray, translation:np.ndarray):
        '''
        rotation: (..., 3, 3)
        '''
        shape_batch = rotation.shape[:-2]
        RT = np.zeros((*shape_batch, 4, 4), dtype=rotation.dtype)
        
        translation = translation.reshape(*shape_batch, 3)
        RT[..., :3, :3] = rotation
        RT[..., :3, 3] = translation
        RT[..., 3, 3] = 1
        return RT
    
    
    @staticmethod
    def get_part_pose_from_status(parent_pose, 
                                joint_pivot, 
                                joint_axis, 
                                joint_status,
                                joint_type):
        """
        get the part pose from the parent pose and joint status.

        Args:
            parent_pose (np.ndarray): parent pose, shape (4, 4).
            joint_pivot (np.ndarray): joint pivot point, shape (3,).
            joint_axis (np.ndarray): joint axis, shape (3,).
            joint_status (list): joint status (angle).

        Returns:
            np.ndarray: part pose (4, 4).    
        """

        part_poses = np.zeros((4, 4), dtype=np.float32)



        if joint_status == 0:
            part_poses = parent_pose
        else:
            
            if joint_type == 'revolute':
                axis_angle = joint_axis * joint_status
                joint_matrix = Rt.from_rotvec(axis_angle).as_matrix()
                rmat = np.eye(4, dtype=np.float32)
                rmat[:3, :3] = joint_matrix
            elif joint_type == 'prismatic':
                t = joint_axis * joint_status
                joint_matrix = np.eye(4, dtype=np.float32)
                joint_matrix[:3, 3] = t
                rmat = joint_matrix
            else:
                raise ValueError(f"Unsupported joint type: {joint_type}")
            # rotation_matrix = (axis_angle_to_matrix(torch.from_numpy(axis_angle))).detach().cpu().numpy()
            
            tmat1 = np.eye(4, dtype=np.float32)
            tmat1[:3, 3] = -joint_pivot
            tmat2 = np.eye(4, dtype=np.float32)
            tmat2[:3, 3] = joint_pivot

            
            part_poses = parent_pose @ tmat2 @ rmat @ tmat1

        return part_poses
    
    @staticmethod
    def get_status_from_part_pose(parent_pose, 
                                part_pose, 
                                joint_pivot, 
                                joint_axis, 
                                joint_type):
        """
        get the joint status from the parent part pose and current part pose.
        
        Args:
            parent_pose (np.ndarray): parent part pose, shape (4, 4).
            part_pose (np.ndarray): current part pose, shape (4, 4).
            joint_pivot (np.ndarray): joint pivot point, shape (3,).
            joint_axis (np.ndarray): joint axis, shape (3,).
            joint_type (str): joint type ('revolute' or 'prismatic').

        Returns:
            float: joint status
        """        
        
        delta_transform = np.linalg.inv(parent_pose) @ part_pose
        
        if joint_type == 'revolute':
            axis_angle = Rt.from_matrix(delta_transform[:3, :3]).as_rotvec()
            status = np.linalg.norm(axis_angle, axis=-1)
            axis = axis_angle / status
            
            if HOI4D.angle_between_vectors(axis, joint_axis) > 1.57:
                status = -status
                
        elif joint_type == 'prismatic':
            status = delta_transform[:3, 3] * joint_axis
            status = np.linalg.norm(status, axis=-1)
            
            axis = delta_transform[:3, 3] / status
            
            if HOI4D.angle_between_vectors(axis, joint_axis) > 1.57:
                status = -status
                
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")
            
        return status
    
    
    @staticmethod
    def angle_between_vectors(v1:np.ndarray, v2:np.ndarray, degree=False):
        nv1 = v1 / np.linalg.norm(v1)
        nv2 = v2 / np.linalg.norm(v2)
        rad = np.arccos(np.clip(np.dot(nv1, nv2), -1.0, 1.0))
        if degree:
            return rad / np.pi * 180.0
        else:
            return rad

    @staticmethod
    def remove_small_noise_from_mask(
        mask: np.ndarray, 
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Removes small, isolated noise from a binary mask using morphological opening.

        This function is effective at cleaning up masks that have small, unwanted
        patches of pixels, often resulting from segmentation processes.

        Args:
            mask (np.ndarray): The input binary mask. It can be a boolean array or
                            a uint8 array (where 0 is background and non-zero is foreground).
            kernel_size (int, optional): The size of the square kernel used for the
                                        morphological operation. A larger kernel will
                                        remove larger noise. Must be an odd number.
                                        Defaults to 3.

        Returns:
            np.ndarray: The cleaned binary mask, with the same type as the input.
        """
        # --- 1. Input Validation and Preparation ---
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number.")

        # Ensure the mask is in uint8 format for OpenCV functions,
        # as boolean arrays are not directly supported by morphologyEx.
        # We store the original dtype to convert back at the end.
        original_dtype = mask.dtype
        if original_dtype == bool:
            # Convert boolean (True/False) to uint8 (255/0)
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask.copy()

        # --- 2. Define the Morphological Kernel ---
        # A square kernel is created. The size determines the aggressiveness of the noise removal.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # --- 3. Apply Morphological Opening ---
        # cv2.MORPH_OPEN performs an erosion followed by a dilation.
        cleaned_mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # --- 4. Convert Back to Original Data Type ---
        if original_dtype == bool:
            # Convert back to boolean (non-zero pixels become True)
            cleaned_mask = cleaned_mask_uint8.astype(bool)
        else:
            cleaned_mask = cleaned_mask_uint8
            
        return cleaned_mask

    @staticmethod
    def find_mask_islands(
        mask: np.ndarray, 
        min_area: int = 10
    ) -> list:
        """
        Finds all connected components (islands) in a binary mask and returns their properties.

        This function uses OpenCV's connected components analysis to identify separate
        blobs of pixels in a mask and extracts their size, bounding box, and centroid.

        Args:
            mask (np.ndarray): The input binary mask. It can be a boolean array or
                            a uint8 array (where 0 is background and non-zero is foreground).
            min_area (int, optional): The minimum pixel area for an island to be included
                                    in the results. This is useful for filtering out small noise.
                                    Defaults to 10.

        Returns:
            list: A list of dictionaries. Each dictionary represents one found island
                and contains the following keys:
                - 'id' (int): The label ID of the island.
                - 'size' (int): The area of the island in pixels.
                - 'bbox' (tuple): The bounding box as (x, y, width, height).
                - 'centroid' (tuple): The center of the island as (x, y).
        """
        # --- 1. Ensure the mask is in the correct format (uint8) ---
        if mask.dtype == bool:
            # Convert boolean mask (True/False) to uint8 (255/0)
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask.copy()

        # --- 2. Perform connected components analysis ---
        # The function returns the total number of labels, a labeled image,
        # stats for each label, and the centroids.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, 
            connectivity=8,  # Use 8-way connectivity (considers diagonal pixels as connected)
            ltype=cv2.CV_32S
        )

        # --- 3. Process the results for each island ---
        islands = []
        # We start the loop from 1 because label 0 is always the background.
        for i in range(1, num_labels):
            # Extract stats for the current island
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out islands smaller than the minimum area
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Get the centroid
                cx, cy = centroids[i]

                islands.append({
                    'id': i,
                    'size': area,
                    'bbox': (x, y, w, h),
                    'centroid': (int(cx), int(cy))
                })
                
        return islands


    @staticmethod
    def remove_islands(isl:list, mask:np.ndarray, max_size=2000) -> np.ndarray:
        """
        Remove islands from the mask based on the provided island list.

        Args:
            isl (list): List of islands to be removed.
            mask (np.ndarray): The input binary mask.

        Returns:
            np.ndarray: The mask with specified islands removed.
        """
        biggest_area = 0
        biggest_x = None
        biggest_y = None
        for island in isl:
            if island['size'] > biggest_area:
                biggest_area = island['size']
                biggest_x = island['centroid'][0]
                biggest_y = island['centroid'][1]

        for island in isl:
            x, y = island['centroid']
            bbox = island['bbox'] # (x(row), y(col), width, height)
            dis_2_biggest = np.sqrt((x - biggest_x) ** 2 + (y - biggest_y) ** 2)
            if (dis_2_biggest > 200 and island['size'] < max_size) or (island['size'] < 100):
                # Remove the island from the mask in place
                mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 0
                
        return mask
    
    @staticmethod
    def fps(points, n_samples):
        """
        Farthest Point Sampling
        
        Args:
            points (np.ndarray): 输入点云，形状为 (N, 3)
            n_samples (int): 采样点的数量
            
        Returns:
            tuple: (采样后的点, 采样点的索引)
                - sampled_points (np.ndarray): 采样后的点云，形状为 (n_samples, 3)
                - sampled_indices (np.ndarray): 采样点在原始点云中的索引，形状为 (n_samples,)
        """
        N, _ = points.shape
        
        # 初始化
        sampled_indices = np.zeros(n_samples, dtype=np.int32)
        distances = np.full(N, np.inf)
        
        # 随机选择第一个点
        farthest = np.random.randint(0, N)
        
        for i in range(n_samples):
            sampled_indices[i] = farthest
            
            # 计算当前点到所有点的距离
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            
            # 更新每个点到已选择点集的最小距离
            mask = dist < distances
            distances[mask] = dist[mask]
            
            # 选择距离最远的点作为下一个点
            farthest = np.argmax(distances)
        
        sampled_points = points[sampled_indices]
        return sampled_points, sampled_indices


def dataset_manual_cache(root_dir, cache_dir, category, mode):
    import torch
    import tqdm
    hoi4d = HOI4D(root_dir, sdf_mode=False, category=category, mode=mode, 
                  add_noise=False, debug=False, 
                  force_write_cache=True, use_cache=True, cache_dir=cache_dir)

    assert hoi4d.use_cache, "Dataset is not using cache."
    assert hoi4d.force_write_cache, "force_write_cache is not enabled."
    assert not hoi4d.is_debug, "Debug mode is not supported for manual caching."


    train_loader = torch.utils.data.DataLoader(hoi4d, batch_size=48, shuffle=False, num_workers=48,
                                                        drop_last=False)

    for batch_idx, data in enumerate(tqdm.tqdm(train_loader)):
        ...

def visualize_and_debug(root_dir, cache_dir, out_dir, category, mode):

    hoi4d = HOI4D(root_dir, sdf_mode=False, category=category, mode=mode, 
                  add_noise=False, debug=True, 
                  force_write_cache=False, use_cache=False, cache_dir=cache_dir)
    
    for i in range(len(hoi4d))[::10]:
        data = hoi4d[i]
        print(i)
        with open(f'{out_dir}/tempfile/{i}.pkl', 'wb') as f:
            pickle.dump(data, f)


def get_occlusion_instance(root_dir, cache_dir, out_dir, category):

    hoi4d = HOI4D(root_dir, sdf_mode=False, category=category, mode='val', 
                  add_noise=False, debug=True, 
                  force_write_cache=False, use_cache=False, cache_dir=cache_dir)
    
    for i in range(len(hoi4d))[::10]:
        data = hoi4d[i]
        print(i)
        with open(f'{out_dir}/tempfile/{i}.pkl', 'wb') as f:
            pickle.dump(data, f)


 
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HOI4D dataset helper utilities')
    parser.add_argument('helper', choices=['dataset_manual_cache', 'visualize_and_debug'])
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--out-dir')
    parser.add_argument('--category', required=True)
    parser.add_argument('--mode', choices=['train', 'val', 'test'], required=True)
    args = parser.parse_args()

    if args.helper == 'dataset_manual_cache':
        dataset_manual_cache(args.root_dir, args.cache_dir, args.category, args.mode)
    else:
        if args.out_dir is None:
            parser.error('--out-dir is required for visualize_and_debug')
        visualize_and_debug(args.root_dir, args.cache_dir, args.out_dir, args.category, args.mode)
    
    
    # C14 Train 35100/117 test 5400/18 33 instances for train 4 instances for val
    # C3 Train 38100/127 test 4500/15 51 instances for train 4 instances for val
    # C4 Train 38700/129 test 4200/14 37 instances for train 4 instances for val
    # C6 Train 42300/141 test 5100/17 45 instances for train 4 instances for val
    # Total 154200/514 test 19200/64 166 instances for train 16 instances for val
    # for data in tqdm.tqdm(hoi4d):
    #     ...
    
    # dataset_manual_cache('C3', 'val')
    # import pickle
    
    # for i in range(99, 300):
    #     data = hoi4d.get_frame_with_name('ZY20210800003/H3/C6/N01/S85/s03/T2', i)
    #     with open(f'tempfile/{i}.pkl', 'wb') as f:
    #         pickle.dump(data, f)
    # exit()
   
    
            
    
    
     