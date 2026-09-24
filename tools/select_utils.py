import torch
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

def plot_hist(bar={}, xmark={}, line_x={}, line_y={}, path='./hist.png'):
    # 绘制直方图
    figure = plt.figure(figsize=(15, 15), dpi=150)
    ax = figure.subplots(1, 1)
    if isinstance(bar, dict):
        for k, v in bar.items():
            ax.hist(v, bins=30, density=True, alpha=0.6, label=k, )
    
    # 绘制曲线
    if isinstance(line_y, dict):
        for k, v in line_y.items():
            if k in line_x.keys():
                x = line_x[k]
                ax.plot(x, v, label=k)
    
    # xmin, xmax = plt.xlim()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycler = cycle(color_cycle)
    
    for k, x in xmark.items():
        color = next(color_cycler)
        ax.axvline(x=x, linestyle='--', linewidth=2, label=k, color=color)

    plt.legend()

    plt.savefig(path)
    plt.close()

    




def guassian_pdf(x, mean, std_dev):
    return torch.exp(-((x - mean)**2 / (2 * std_dev**2)))

def calc_dist(pred_observed_global_kp, projected_kp, weight=None, weight_r=0.5):
    '''
    params
        pred_observed_global_kp : B, M, P, KP/P, 3
        projected_kp            : B, N, P, KP/P, 3
        
        weight                  : B, M, P, KP/P, 1
    
    return
        dis     : B, M, N, P, KP/P
        min_v   : B, N, P, KP/P
        min_i   : B, N, P, KP/P
    '''
    dis = torch.linalg.norm(projected_kp.unsqueeze(1) - pred_observed_global_kp.unsqueeze(2), dim=-1)
    
    if weight is not None:
        w = guassian_pdf(weight, 0, weight_r)
        dis *= w.unsqueeze(2).squeeze(-1)
    
    min_v, min_i = dis.min(dim=1)
    
    return dis, min_v, min_i
    
def calc_score(dist, std, mean=0):
    '''
    params
        dist: B, M, N, KP
        std: B, M, N, KP
    
    return
        score: B, M, N
    '''
    
    mean_shape = list(dist.shape)
    mean_shape[1] = 1
    mean = torch.ones(mean_shape, device=dist.device) * mean
    # mean = torch.mean(dist, dim=1, keepdim=True)
    std = torch.ones_like(mean) * std
    # std = torch.std(dist, dim=1, keepdim=True) 
    score = guassian_pdf(dist, mean, std) # B, N, P, KP/P
    score = score.sum(dim=-1)
    return score

def select_by_score(score, pred_rt):
    '''
    score: B, 1, N
    pred_rt: B, S, P, 4, 4
    '''
    max_s, max_i = score.max(dim=-1, keepdim=True) # B, 1, 1
    KPP = pred_rt.shape[-3]
    selected_pose = torch.gather(pred_rt, 1, max_i[..., None, None].repeat(1, 1, KPP, 4, 4))
    return selected_pose, max_i
    

def select_by_id(id, data, shared_dim=2, idx_dim=1):
    '''
    id: B, N, ...
    data: B, S, ...
    '''
    
    data_identi_shape = data.shape[shared_dim:]
    id_identi_shape = id.shape[shared_dim:]
    det_dim = len(data_identi_shape) - len(id_identi_shape)
    
    # make sure the shape[shared_dim:] of id is all 1
    if len(id_identi_shape) != sum(id_identi_shape):
        raise ValueError('id shape error')
    
    if det_dim < 0:
        raise ValueError('id dim error')
    
    # extern the id shape to data shape
    id_ext = id.reshape(*id.shape, *([1]*det_dim))
    
        
    
    selected_data = torch.gather(data, idx_dim, id_ext.repeat(*([1]*shared_dim), *data_identi_shape))
    return selected_data, id
