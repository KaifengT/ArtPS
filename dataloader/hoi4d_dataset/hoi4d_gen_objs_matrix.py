import trimesh
import numpy as np
import time
import os
import pickle

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1
    return  T



root = 'HOI4D/cad_model/articulated'

not_ok_ins = []

cate_path = os.listdir(root)

for cate in cate_path:
    cate_folder = os.path.join(root, cate)
    
    if not os.path.isdir(cate_folder):
        continue
    ins_list = os.listdir(cate_folder)
    
    
    for ins in ins_list:
        instance_folder = os.path.join(cate_folder, ins)
        try:
            obj_paths = os.listdir(os.path.join(instance_folder, 'objs'))
        except Exception as e:
            continue
        
        for obj_path in obj_paths:
            if 'align' in obj_path:
                continue
            
            a_obj_abs_path = os.path.join(instance_folder, 'objs', obj_path)
            obj_name = os.path.basename(a_obj_abs_path)
            obj_file_name, obj_ext = os.path.splitext(obj_name)
            
            if os.path.exists(os.path.join(instance_folder, 'objs', obj_file_name + '-align-t.npy')):
                print('delete the file: ', os.path.join(instance_folder, 'objs', obj_file_name + '-align-t.npy'))
                os.remove(os.path.join(instance_folder, 'objs', obj_file_name + '-align-t.npy'))
        
            b_obj_abs_path = os.path.join(instance_folder, 'objs', obj_file_name + '-align' + obj_ext)
            
            if not (os.path.exists(b_obj_abs_path) and os.path.isfile(b_obj_abs_path)):
                print(f"Cannot find the file: {b_obj_abs_path}")
                not_ok_ins.append(instance_folder)
                continue
        
            mesh_a = trimesh.load(a_obj_abs_path, process=False)
            mesh_b = trimesh.load(b_obj_abs_path, process=False)
            
            
            mesh_a_vertex = np.array(mesh_a.vertices)
            mesh_b_vertex = np.array(mesh_b.vertices)
            
            try:
                t = best_fit_transform(mesh_a_vertex, mesh_b_vertex)
            except Exception as e:
                print(f"Error processing {obj_name} in {instance_folder}: {e}")
                not_ok_ins.append(instance_folder)
                continue

            print(f"Processing {obj_name} in {instance_folder}...")
            
            tname = os.path.join(instance_folder, 'objs', obj_file_name + '-align-t.npy')
            np.save(tname, t)


np.savetxt('not_ok_ins.txt', not_ok_ins, fmt='%s')