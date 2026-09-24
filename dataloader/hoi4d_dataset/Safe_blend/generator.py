import os
import trimesh
import numpy as np

class Generator:
    def __init__(self, morph_num=20, root_dir:str=''):
        self.base = trimesh.load(os.path.join(root_dir, 'safe.obj'), process=False)

        self.base_vertices = np.array(self.base.vertices)
        self.base_faces = np.array(self.base.faces)
        self.morph_num = morph_num
        self.all_morphs = []
        for i in range(morph_num):
            self.all_morphs.append(self.get_blend_shape(os.path.join(root_dir, f'safe_morph_{i+1:02d}.obj'), self.base_vertices))
            
        self.weightFunc = lambda x: x * 1.5 - 0.5
        self.part_mask = np.load(os.path.join(root_dir, 'mask.npy'))
        assert len(self.part_mask) == len(self.base_vertices), "Mask length must match number of base vertices."

    @staticmethod
    def get_blend_shape(path, base):
        morph = trimesh.load(path, process=False)
        morph_vertices = np.array(morph.vertices)
        offset = morph_vertices - base
        return offset
         
    def _get_bboxs(self, vertices):
        extents = []
        transforms = []
        unique_parts = np.unique(self.part_mask)
        
        for part_id in unique_parts:
            part_verts = vertices[self.part_mask == part_id]
            if len(part_verts) > 0:
                bbox = trimesh.PointCloud(part_verts).bounding_box
                extents.append(bbox.extents)
                transforms.append(bbox.transform)
                
        return np.stack(extents, axis=0, dtype=np.float32), np.stack(transforms, axis=0, dtype=np.float32)
    
    @staticmethod
    def _get_parts(vertices, part_mask):
        parts = []
        unique_parts = np.unique(part_mask)
        
        for part_id in unique_parts:
            part_verts = vertices[part_mask == part_id]
            if len(part_verts) > 0:
                parts.append(trimesh.PointCloud(part_verts))
            
        return parts
            
    def weightFunc(self, func:callable,):
        self.weightFunc = func
        
        
    def _getJoint(self, vertices):
        ref_vertex = vertices[[6646, 7140]]
        ref_vertex2 = vertices[10501]
        joint_pivots = np.mean(ref_vertex, axis=0)
        joint_pivots[2] = 0.0
        joint_pivots[0] = ref_vertex2[0]
        joint_pivots = np.stack([np.array([0., 0., 0.]), joint_pivots], dtype=np.float32)
        joint_axis = np.array([[0., 0., 0.], [0.0, 0.0, 1.0]], dtype=np.float32)
        return joint_pivots, joint_axis
        
    def generate(self, morph_values, noise:float=0.003):
        assert len(morph_values) == len(self.all_morphs), "Morph values length must match number of morphs. needed: {}, got: {}".format(len(self.all_morphs), len(morph_values))
        
        blended_vertices = np.array(self.base_vertices)
        for i, weight in enumerate(morph_values):
            weight = self.weightFunc(weight)
            if weight < 0:
                weight = 0.0
            blended_vertices += self.all_morphs[i] * weight

        joint_pivots, joint_axis = self._getJoint(blended_vertices)
        parts = self._get_parts(blended_vertices, self.part_mask)
        
        blended_vertices += np.random.randn(*blended_vertices.shape) * noise
        
        obj_data = {
            'one_mesh': trimesh.Trimesh(vertices=blended_vertices, faces=self.base_faces, process=False),
            'joint_pivots_matrix': joint_pivots,
            'joint_axis_matrix': joint_axis,
            'preload_meshs': parts,
        }

        return obj_data
