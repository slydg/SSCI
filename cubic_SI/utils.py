import torch
import numpy as np
import ot
import math
from tqdm import tqdm

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

def compute_emd2(ref_data, pred_data, p=2):
    M = torch.cdist(pred_data, ref_data, p=p)
    a, b = ot.unif(pred_data.size()[0]), ot.unif(ref_data.size()[0])
    loss = ot.emd2(a, b, M.cpu().detach().numpy(),  numItermax=1000000)
    return loss

def marginal_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :]
        pred_traj = pred_traj[:, eval_idx, :]
        int_time = int_time[eval_idx]
    
    if pred_traj.ndim == 3:
        data_size, t_size, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            ref_dist = ref_traj[:, j]
            pred_dist = pred_traj[:, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            res[f't={int_time[j].item()}'] = { 'mean' : loss }
        
        return res

    elif pred_traj.ndim == 4:
        data_size, t_size, num_repeat, dim = pred_traj.size()
        res = {}
        for j in range(1, t_size):
            losses = []
            for i in range(num_repeat):
                ref_dist = ref_traj[:, j]
                pred_dist = pred_traj[:, j, i]
                M = torch.cdist(ref_dist, pred_dist, p=p)
                a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
                loss = ot.emd2(a, b, M.cpu().detach().numpy())
                losses.append(loss)
            
            res[f't={int_time[j].item()}'] = { 'mean' : np.mean(losses), 'std' : np.std(losses) }
        return res
    
def conditional_distribution_discrepancy(ref_traj, pred_traj, int_time, eval_idx=None, p=2):
    if not eval_idx is None:
        ref_traj = ref_traj[:, eval_idx, :, :]
        pred_traj = pred_traj[:, eval_idx, :, :]
        int_time = int_time[eval_idx]

    data_size, t_size, num_repeat, dim = ref_traj.size()
    res = {}
    for j in range(1, t_size):
        losses = []
        for i in range(data_size):
            ref_dist = ref_traj[i, j]
            pred_dist = pred_traj[i, j]
            M = torch.cdist(ref_dist, pred_dist, p=p)
            a, b = ot.unif(ref_dist.size()[0]), ot.unif(pred_dist.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            losses.append(loss)
        res[f't={int_time[j].item()}'] = sum(losses) / data_size
    return res

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    batch_size = 200
    num_window = int(total0.shape[0]/batch_size)+1
    L2_dis = []
    for i in tqdm(range(num_window)):
        diff = (total0[i*batch_size:(i+1)*batch_size].cuda()-total1[i*batch_size:(i+1)*batch_size].cuda())
        diff.square_()
        L2_dis.append(diff.sum(2).cpu())
    L2_distance = torch.concatenate(L2_dis,dim=0)


    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss