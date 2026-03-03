import torch
import numpy as np
from scipy.interpolate import CubicSpline
try:
    import torchcubicspline
    TORCHCUBSPLINE_AVAILABLE = True
except ImportError:
    TORCHCUBSPLINE_AVAILABLE = False
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

import ot
import warnings


from typing import Union



def Interp_t_linear(timepoint_lists_for_each_stage,x_0,x_1,stage_index,alpha=None,beta=None):
    B=x_0.shape[0]
    d=x_0.shape[1]
    t=torch.from_numpy(np.random.choice(timepoint_lists_for_each_stage[stage_index],B)).reshape(-1,1).cuda()
    # Z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=1, size=d) for i in range(B)])).cuda()
    t_ceil = timepoint_lists_for_each_stage[stage_index][-1]
    t_floor = timepoint_lists_for_each_stage[stage_index][0]
    # x_t=((t_ceil-t)/(t_ceil-t_floor))*x_0+((t-t_floor)/(t_ceil-t_floor))*x_1+torch.sqrt(u*((t-t_floor)/(t_ceil-t_floor))*((t_ceil-t)/(t_ceil-t_floor)))*Z
    if alpha is not None and beta is not None:
        x_t=alpha(t)*x_0+beta(t)*x_1
    else:
        x_t=((t_ceil-t)/(t_ceil-t_floor))*x_0+((t-t_floor)/(t_ceil-t_floor))*x_1
    return x_t,t,t_ceil,t_floor

















def generate_one_stage(stage_id,time_stu,x_start,b_m,kappa_m,delta_t,u,gamma,SDE=True,conditions=None,dynamic_conditions=False):
    B=x_start.shape[0]
    d=x_start.shape[1]
    path=[]
    path.append(x_start.detach().cpu())
    if conditions is not None and dynamic_conditions:
        conditions = conditions[:,stage_id,:]
    for t_new in time_stu[:-1]:
        t=torch.from_numpy(np.repeat(t_new,B)).reshape(-1,1).float().cuda()
        x_t=path[-1].cuda().float()
        
        if not SDE:
            del_x_t=b_m(x_t.float(),t.float(),conditions=conditions.cuda().float())*delta_t
            path.append((x_t+del_x_t.cuda()).detach().cpu())
        else:
            z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=delta_t, size=d) for i in range(B)])).cuda()
            if t_new == time_stu[0]:
                del_x_t=b_m(x_t.float(),t.float(),conditions=conditions.cuda().float())*delta_t #+ u(t) * kappa_m(x_t.float(),t.float(),conditions=conditions.cuda().float())*delta_t + np.sqrt(2*u(t))*z

            else:
                del_x_t=b_m(x_t.float(),t.float(),conditions=conditions.cuda().float())*delta_t + (u(t)/gamma(t)) * kappa_m(x_t.float(),t.float(),conditions=conditions.cuda().float())*delta_t + np.sqrt(2*u(t))*z

            path.append((x_t+del_x_t.cuda()).detach().cpu())
    return path



def generate_path(timepoint_lists_for_each_stage,x_start,b_m,kappa_m,delta_t,u,gamma,SDE=True,conditions=None,conditional_path_encoder=None,hist_times=None,dynamic_conditions=False):
    path = []
    stage_index_list = list(range(len(timepoint_lists_for_each_stage)))
    path.append(x_start.detach().cpu())
    if conditional_path_encoder is not None:
        conditions = conditional_path_encoder(conditions.float().cuda(),torch.tensor(hist_times).repeat(conditions.shape[0],1).float().cuda())
    for stage_id in stage_index_list:
        path += generate_one_stage(stage_id,timepoint_lists_for_each_stage[stage_id],path[-1],b_m,kappa_m,delta_t,u,gamma,SDE=SDE,conditions=conditions,dynamic_conditions=dynamic_conditions)[1:]
    return path







def gamma_function(
    t: torch.Tensor,
    zero_times: torch.Tensor,
    func_type: str = 'sqrt'
) -> torch.Tensor:
    """
    计算一个在给定`zero_times`上为0的非负函数。
    提供了多种函数类型，其中一些是处处可微的。

    Args:
        t (torch.Tensor): 需要计算函数值的时间点张量 (批处理)。
        zero_times (torch.Tensor): 函数值为0的时刻点的一维张量。
                                   重要: 此张量必须预先按升序排序。
        func_type (str): 要使用的函数类型。可选值为:
                         'sqrt': 原始函数 constant * sqrt((t - t_0) * (t_1 - t) / (t_1 - t_0)) (假设 constant=1)。
                                 在 zero_times 点不可微。
                         'sine': 基于 sin^2 的函数，在 zero_times 点可微 (C¹连续)。
                         'poly': 基于 (t-t_0)^2 * (t_1-t)^2 的四次多项式函数，
                                 在 zero_times 点可微 (C¹连续)。
                         'sine_corner': 基于 sin 的函数，在 zero_times 点具有有限的左/右导数（尖角）。
                         'poly_corner': 基于 (t-t_0)*(t_1-t) 的二次多项式函数，在 zero_times 点具有有限的左/右导数（尖角）。
                         'None': 返回一个全零的函数。
    Returns:
        torch.Tensor: 与输入t形状相同的函数值张量。
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("输入 t 必须是 PyTorch 张量。")
    if not isinstance(zero_times, torch.Tensor):
        raise TypeError("输入 zero_times 必须是 PyTorch 张量。")
    if func_type not in ['sqrt', 'sine', 'poly', 'sine_corner', 'poly_corner','None','extreme','small']:
        raise ValueError("func_type 必须是 'sqrt', 'sine', 'poly', 'sine_corner', 或 'poly_corner' 之一。")

    device = t.device
    dtype = t.dtype
    _zero_times = zero_times.to(device=device, dtype=dtype).clone().detach()
    y = torch.zeros_like(t, device=device, dtype=dtype)

    if _zero_times.numel() < 2:
        return y
    
    if func_type == 'None':
        return y
    
    
    for i in range(_zero_times.numel() - 1):
        t0 = _zero_times[i]
        t1 = _zero_times[i+1]

        if t0 == t1:
            continue

        mask = torch.logical_and(t > t0, t < t1)
        t_interval = t[mask]

        if t_interval.numel() == 0:
            continue

        y_values_in_interval = torch.zeros_like(t_interval)
        
        # Based on the chosen function type, calculate the values
        if func_type == 'sqrt':
            term_under_sqrt = (t_interval - t0) * (t1 - t_interval) / (t1 - t0)
            # clamp is for numerical stability
            y_values_in_interval = torch.sqrt(torch.clamp(term_under_sqrt, min=0.0))
            
        elif func_type == 'extreme':
            term_under_sqrt = (t_interval - t0) * (t1 - t_interval) / (t1 - t0)
            # clamp is for numerical stability
            y_values_in_interval = 10 * torch.sqrt(torch.clamp(term_under_sqrt, min=0.0))

        elif func_type == 'small':
            # This function is C¹ continuous. Derivative is 0 at t0 and t1.
            normalized_time = (t_interval - t0) / (t1 - t0)
            y_values_in_interval = 1e-4 * torch.sin(torch.pi * normalized_time)**2

        elif func_type == 'sine':
            # This function is C¹ continuous. Derivative is 0 at t0 and t1.
            normalized_time = (t_interval - t0) / (t1 - t0)
            y_values_in_interval = torch.sin(torch.pi * normalized_time)**2
            
        elif func_type == 'poly':
            # This function is also C¹ continuous. Derivative is 0 at t0 and t1.
            # (t - t0)^2 * (t1 - t)^2.
            # The scaling factor 16 / (t1 - t0)^4 normalizes the peak to 1.
            # We can omit it if normalization is not required.
            normalized_val = 16.0 / ((t1 - t0)**4)
            y_values_in_interval = normalized_val * ((t_interval - t0)**2) * ((t1 - t_interval)**2)
        
        elif func_type == 'sine_corner':
            # Has finite left/right derivatives at t0 and t1, creating a "corner".
            normalized_time = (t_interval - t0) / (t1 - t0)
            # sin is non-negative for inputs from 0 to pi.
            y_values_in_interval = torch.sin(torch.pi * normalized_time)

        elif func_type == 'poly_corner':
            # A simple parabola. Has finite left/right derivatives at t0 and t1.
            # Scaling by 4 / (t1 - t0)**2 normalizes the peak value to 1.
            normalized_val = 4.0 / ((t1 - t0)**2)
            y_values_in_interval = normalized_val * (t_interval - t0) * (t1 - t_interval)

        y[mask] = y_values_in_interval
            
    return y


def gamma_derivative(
    t: torch.Tensor,
    zero_times: torch.Tensor,
    func_type: str = 'sqrt'
) -> torch.Tensor:
    """
    显式计算 gamma_function 相对于 t 的解析导数。

    Args:
        t (torch.Tensor): 需要计算导数值的时间点张量 (批处理)。
        zero_times (torch.Tensor): 函数值为0的时刻点的一维张量。
                                   重要: 此张量必须预先按升序排序。
        func_type (str): 使用的函数类型。必须与 gamma_function 中使用的一致。
                         可选值: 'sqrt', 'sine', 'poly', 'sine_corner', 'poly_corner'。

    Returns:
        torch.Tensor: 与输入t形状相同的导数值张量。
                      对于 'sqrt', 当 t 逼近 zero_times 点时，导数值会趋向于 +/- 无穷大。
                      对于 'sine' 和 'poly', 在 zero_times 点导数为0。
                      对于 'sine_corner' 和 'poly_corner', 在 zero_times 点具有有限的左/右导数。
                      对于'None', 返回一个全零的函数。
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("输入 t 必须是 PyTorch 张量。")
    if not isinstance(zero_times, torch.Tensor):
        raise TypeError("输入 zero_times 必须是 PyTorch 张量。")
    if func_type not in ['sqrt', 'sine', 'poly', 'sine_corner', 'poly_corner','None','extreme','small']:
        raise ValueError("func_type 必须是 'sqrt', 'sine', 'poly', 'sine_corner', 或 'poly_corner' 之一。")

    device = t.device
    dtype = t.dtype
    _zero_times = zero_times.to(device=device, dtype=dtype).clone().detach()
    df_dt = torch.zeros_like(t, device=device, dtype=dtype)

    if _zero_times.numel() < 2:
        return df_dt
    if func_type == 'None':
        return df_dt
    
    for i in range(_zero_times.numel() - 1):
        t0 = _zero_times[i]
        t1 = _zero_times[i+1]

        if t0 == t1:
            continue

        mask = torch.logical_and(t > t0, t < t1)
        t_interval = t[mask]

        if t_interval.numel() == 0:
            continue

        df_dt_interval = torch.zeros_like(t_interval)

        # Calculate derivative based on the chosen function type
        if func_type == 'sqrt':
            # Derivative: (t0 + t1 - 2t) / (2 * sqrt((t - t0)(t1 - t)(t1 - t0)))
            numerator_val = t0 + t1 - (2.0 * t_interval)
            term_inside_sqrt = (t_interval - t0) * (t1 - t_interval) * (t1 - t0)
            denominator_val = 2.0 * torch.sqrt(term_inside_sqrt)
            # Avoid division by zero at the exact endpoints, though mask should prevent this.
            # PyTorch handles division by zero by returning inf, which is correct here.
            df_dt_interval = numerator_val / denominator_val
            
        elif func_type == 'extreme':
            # Derivative: (t0 + t1 - 2t) / (2 * sqrt((t - t0)(t1 - t)(t1 - t0)))
            numerator_val = t0 + t1 - (2.0 * t_interval)
            term_inside_sqrt = (t_interval - t0) * (t1 - t_interval) * (t1 - t0)
            denominator_val = 2.0 * torch.sqrt(term_inside_sqrt)
            # Avoid division by zero at the exact endpoints, though mask should prevent this.
            # PyTorch handles division by zero by returning inf, which is correct here.
            df_dt_interval = 10 * (numerator_val / denominator_val)

        elif func_type == 'small':
            # Derivative of sin^2(pi * (t-t0)/(t1-t0)) is:
            # pi/(t1-t0) * 2 * sin(...) * cos(...) = pi/(t1-t0) * sin(2*...)
            normalized_time = (t_interval - t0) / (t1 - t0)
            constant_factor = torch.pi / (t1 - t0)
            df_dt_interval = 1e-4 * constant_factor * torch.sin(2.0 * torch.pi * normalized_time)      
            
        elif func_type == 'sine':
            # Derivative of sin^2(pi * (t-t0)/(t1-t0)) is:
            # pi/(t1-t0) * 2 * sin(...) * cos(...) = pi/(t1-t0) * sin(2*...)
            normalized_time = (t_interval - t0) / (t1 - t0)
            constant_factor = torch.pi / (t1 - t0)
            df_dt_interval = constant_factor * torch.sin(2.0 * torch.pi * normalized_time)

        elif func_type == 'poly':
            # Derivative of 16/(t1-t0)^4 * (t-t0)^2 * (t1-t)^2 is:
            # 32/(t1-t0)^4 * (t-t0)(t1-t) * (t0+t1-2t)
            constant_factor = 32.0 / ((t1 - t0)**4)
            term1 = t_interval - t0
            term2 = t1 - t_interval
            term3 = t0 + t1 - 2.0 * t_interval
            df_dt_interval = constant_factor * term1 * term2 * term3

        elif func_type == 'sine_corner':
            # Derivative of sin(pi * (t-t0)/(t1-t0)) is:
            # pi/(t1-t0) * cos(pi * (t-t0)/(t1-t0))
            normalized_time = (t_interval - t0) / (t1 - t0)
            constant_factor = torch.pi / (t1 - t0)
            df_dt_interval = constant_factor * torch.cos(torch.pi * normalized_time)

        elif func_type == 'poly_corner':
            # Derivative of 4/(t1-t0)^2 * (t-t0)(t1-t) is:
            # 4/(t1-t0)^2 * (t0+t1-2t)
            constant_factor = 4.0 / ((t1 - t0)**2)
            df_dt_interval = constant_factor * (t0 + t1 - 2.0 * t_interval)

        df_dt[mask] = df_dt_interval
            
    return df_dt







def compute_dct_from_4d(windows_4d, dct_matrix):
    """
    Args:
        windows_4d: (Batch, Time, Channels, Window) 
                    这是 DataLoader 直接吐出来的 4D 张量
        dct_matrix: (Window, Window)
    
    Returns:
        dct_features: (Batch, Time, Channels * Window)
        用于计算 Loss
    """
    # Move matrix to same device
    M = dct_matrix.to(windows_4d.device)
    
    # Operation: Window vector @ Matrix.T
    # (B, T, C, W) @ (W, W) -> (B, T, C, W)
    dct_coeffs = torch.matmul(windows_4d, M.t())
    
    # Flatten last two dims
    B, T, C, W = dct_coeffs.shape
    return dct_coeffs.reshape(B, T, C * W)


def inverse_dct_to_raw(dct_features, dct_matrix, n_channels):
    """
    Inference 时使用：从预测的 DCT 系数还原当前时刻电压
    Args:
        dct_features: (Batch, Time, Channels * Window)
    Returns:
        raw_current: (Batch, Time, Channels)
    """
    B, T, _ = dct_features.shape
    W = dct_matrix.shape[0]
    C = n_channels
    M_inv = dct_matrix.to(dct_features.device).t() # IDCT Matrix
    
    # Reshape
    dct_coeffs = dct_features.view(B, T, C, W)
    
    # Inverse Project
    window_rec = torch.matmul(dct_coeffs, M_inv.t())
    
    # 只取窗口的最后一个点作为当前电压
    return window_rec[..., -1]













# 尝试导入 torchcubicspline
try:
    from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
    torchcubicspline_available = True
    print("Successfully imported configured torchcubicspline.")
except ImportError:
    torchcubicspline_available = False
    print("WARNING: torchcubicspline library not found or configured as expected.")
    print("Cubic spline functions requiring it will not work and will raise an error if called.")
    class NaturalCubicSpline:
        def __init__(self, coeffs): pass
        def evaluate(self, t_eval, extrapolate=False): 
            raise ImportError("torchcubicspline not found, cannot evaluate spline.")
        def derivative(self, t_eval, extrapolate=False): 
            raise ImportError("torchcubicspline not found, cannot evaluate spline derivative.")
    def natural_cubic_spline_coeffs(t, y): 
        raise ImportError("torchcubicspline not found, cannot compute spline coefficients.")

# --- PyTorch Custom Linear Interpolation ---

# MODIFIED: Combined helper function for coefficients and their derivatives w.r.t. u
def get_custom_linear_coeffs_and_derivs_pytorch(u_normalized_pt: torch.Tensor):
    """
    Calculates alpha, beta and their derivatives w.r.t. u based on normalized time u.
    Users modify this function to implement different interpolation schemes.
    
    Current implementation: Standard Linear Interpolation
    alpha(u) = 1 - u  => d(alpha)/du = -1
    beta(u)  = u      => d(beta)/du  =  1
    
    Boundary conditions satisfied:
    alpha(0)=1, beta(0)=0
    alpha(1)=0, beta(1)=1
    """
    # Ensure calculations are done in float for consistency
    u_float = u_normalized_pt.to(torch.promote_types(u_normalized_pt.dtype, torch.float32))
    
    # Coefficients
    alpha_u_pt = 1.0 - u_float
    beta_u_pt = u_float
    
    # Derivatives w.r.t. u
    d_alpha_du_pt = -torch.ones_like(u_float)
    d_beta_du_pt = torch.ones_like(u_float)
    
    # Return matching the input dtype if possible, derivatives remain float
    return (alpha_u_pt.to(u_normalized_pt.dtype), 
            beta_u_pt.to(u_normalized_pt.dtype), 
            d_alpha_du_pt, 
            d_beta_du_pt)

# MODIFIED: Interpolation function now calls the new helper
def batched_tasks_custom_linear_interpolation_pytorch(
        t_known_pt: torch.Tensor,
        x_known_batched_pt: torch.Tensor, # Shape (B, N, D)
        t_query_pt: torch.Tensor):
    device = x_known_batched_pt.device
    dtype = x_known_batched_pt.dtype

    # --- Input validation and setup (identical to previous) ---
    if not torch.all(torch.diff(t_known_pt) > 0): raise ValueError("t_known_pt sorted check")
    if x_known_batched_pt.ndim != 3: raise ValueError("x_known_batched_pt ndim check")
    B, N, D_val = x_known_batched_pt.shape
    if t_known_pt.shape[0] != N: raise ValueError("N mismatch")
    if t_query_pt.ndim == 1:
        M = t_query_pt.shape[0]
        t_query_eff_pt = t_query_pt.unsqueeze(0).expand(B, M)
    elif t_query_pt.ndim == 2:
        if t_query_pt.shape[0] != B: raise ValueError("B mismatch t_query")
        M = t_query_pt.shape[1]
        t_query_eff_pt = t_query_pt
    else: raise ValueError("t_query_pt ndim check")
    if N < 2: 
        interpolated_x_pt = torch.empty((B, M, D_val), device=device, dtype=dtype)
        if N == 1: interpolated_x_pt[:] = x_known_batched_pt[:, 0:1, :].expand(-1,M,-1)
        else: interpolated_x_pt[:] = float('nan')
        return interpolated_x_pt
    # --- End Input validation and setup ---

    _idx_i_pt = torch.searchsorted(t_known_pt, t_query_eff_pt, right=True) - 1
    _idx_i_pt = torch.clamp(_idx_i_pt, 0, N - 2) 

    t_i_eff_pt = t_known_pt[_idx_i_pt]
    t_i_plus_1_eff_pt = t_known_pt[_idx_i_pt + 1]
    
    batch_indices_for_gather = torch.arange(B, device=device).unsqueeze(1)
    x_i_pt = x_known_batched_pt[batch_indices_for_gather, _idx_i_pt]
    x_i_plus_1_pt = x_known_batched_pt[batch_indices_for_gather, _idx_i_pt + 1]

    delta_t_interval_eff_pt = t_i_plus_1_eff_pt - t_i_eff_pt
    u_calc_dtype = torch.promote_types(dtype, torch.float32)
    u_normalized_pt = torch.zeros_like(delta_t_interval_eff_pt, device=device, dtype=u_calc_dtype)
    
    safe_mask_pt = delta_t_interval_eff_pt > 1e-9 
    u_normalized_pt[safe_mask_pt] = (t_query_eff_pt[safe_mask_pt].to(u_calc_dtype) - t_i_eff_pt[safe_mask_pt].to(u_calc_dtype)) / \
                                    delta_t_interval_eff_pt[safe_mask_pt].to(u_calc_dtype)
    
    u_clamped_for_coeffs = torch.clamp(u_normalized_pt, 0.0, 1.0)
    
    # Call the new helper function, only need alpha and beta here
    alpha_u_pt, beta_u_pt, _, _ = get_custom_linear_coeffs_and_derivs_pytorch(u_clamped_for_coeffs) 
    
    # Ensure interpolation calculation uses compatible types
    interpolated_x_pt = alpha_u_pt.unsqueeze(-1) * x_i_pt.to(alpha_u_pt.dtype) + \
                        beta_u_pt.unsqueeze(-1) * x_i_plus_1_pt.to(beta_u_pt.dtype)
    interpolated_x_pt = interpolated_x_pt.to(dtype) # Cast back to original input data type

    # --- Boundary clamping (identical to previous) ---
    if N > 0 : 
        val_at_abs_start_pt = x_known_batched_pt[:, 0, :].to(dtype)  
        val_at_abs_end_pt = x_known_batched_pt[:, N-1, :].to(dtype) 
        mask_below_start_pt = (t_query_eff_pt < t_known_pt[0]).unsqueeze(-1)
        mask_above_end_pt = (t_query_eff_pt > t_known_pt[-1]).unsqueeze(-1)
        val_at_abs_start_expanded_pt = val_at_abs_start_pt.unsqueeze(1).expand(-1,M,-1)
        val_at_abs_end_expanded_pt = val_at_abs_end_pt.unsqueeze(1).expand(-1,M,-1)
        interpolated_x_pt = torch.where(mask_below_start_pt, val_at_abs_start_expanded_pt, interpolated_x_pt)
        interpolated_x_pt = torch.where(mask_above_end_pt, val_at_abs_end_expanded_pt, interpolated_x_pt)
    # --- End Boundary clamping ---
    
    return interpolated_x_pt

# MODIFIED: Derivative function now uses the general formula and the new helper
def batched_tasks_custom_linear_interpolation_derivative_pytorch(
        t_known_pt: torch.Tensor,
        x_known_batched_pt: torch.Tensor, # Shape (B, N, D)
        t_query_pt: torch.Tensor):
    device = x_known_batched_pt.device
    output_dtype = torch.promote_types(x_known_batched_pt.dtype, torch.float32) # Derivatives are float

    # --- Input validation and setup (identical to previous) ---
    if not torch.all(torch.diff(t_known_pt) > 0): raise ValueError("t_known_pt sorted check")
    if x_known_batched_pt.ndim != 3: raise ValueError("x_known_batched_pt ndim check")
    B, N, D_val = x_known_batched_pt.shape
    if t_known_pt.shape[0] != N: raise ValueError("N mismatch")
    if t_query_pt.ndim == 1:
        M = t_query_pt.shape[0]
        t_query_eff_pt = t_query_pt.unsqueeze(0).expand(B, M)
    elif t_query_pt.ndim == 2:
        if t_query_pt.shape[0] != B: raise ValueError("B mismatch t_query")
        M = t_query_pt.shape[1]
        t_query_eff_pt = t_query_pt
    else: raise ValueError("t_query_pt ndim check")
    # --- End Input validation and setup ---

    derivatives_pt = torch.zeros((B, M, D_val), device=device, dtype=output_dtype)

    if N < 2: 
        return derivatives_pt

    _idx_i_pt = torch.searchsorted(t_known_pt, t_query_eff_pt, right=True) - 1
    # Use clamped indices to determine the relevant interval for derivative calculation
    # Queries outside the range will use the derivative from the first/last segment
    _idx_i_pt = torch.clamp(_idx_i_pt, 0, N - 2)

    t_i_eff_pt = t_known_pt[_idx_i_pt]
    t_i_plus_1_eff_pt = t_known_pt[_idx_i_pt + 1]

    batch_indices_for_gather = torch.arange(B, device=device).unsqueeze(1)
    x_i_pt = x_known_batched_pt[batch_indices_for_gather, _idx_i_pt]
    x_i_plus_1_pt = x_known_batched_pt[batch_indices_for_gather, _idx_i_pt + 1]

    delta_t_interval_eff_pt = (t_i_plus_1_eff_pt - t_i_eff_pt).to(output_dtype)
    
    # Calculate u (normalized time) for the interval
    u_calc_dtype = output_dtype
    u_normalized_pt = torch.zeros_like(delta_t_interval_eff_pt, device=device, dtype=u_calc_dtype)
    safe_mask_pt = delta_t_interval_eff_pt > 1e-9 
    # Note: u is needed by the derivative helper, calculate it based on the *actual* query time
    u_normalized_pt[safe_mask_pt] = (t_query_eff_pt[safe_mask_pt].to(u_calc_dtype) - t_i_eff_pt[safe_mask_pt].to(u_calc_dtype)) / \
                                    delta_t_interval_eff_pt[safe_mask_pt]
    # Unlike interpolation coeffs, derivatives might depend on u outside [0,1] if extrapolating slope
    # We use the u corresponding to the actual t_query_eff_pt for derivatives.

    # Call the new helper function to get d(alpha)/du and d(beta)/du
    _, _, d_alpha_du_pt, d_beta_du_pt = get_custom_linear_coeffs_and_derivs_pytorch(u_normalized_pt) 
    
    # Calculate the term in the parenthesis of the general derivative formula
    # numerator_term = (d(alpha)/du * x_i + d(beta)/du * x_{i+1})
    numerator_term_pt = d_alpha_du_pt.unsqueeze(-1) * x_i_pt.to(output_dtype) + \
                        d_beta_du_pt.unsqueeze(-1) * x_i_plus_1_pt.to(output_dtype)
                        
    # Denominator term is (t_{i+1} - t_i)
    denominator_pt = delta_t_interval_eff_pt.unsqueeze(-1) # Shape (B, M, 1)
    
    # Mask for valid intervals (denominator not zero)
    valid_intervalkappa_mask_pt = (delta_t_interval_eff_pt > 1e-9).unsqueeze(-1) # Shape (B, M, 1)

    # dy/dt = numerator_term / denominator = numerator_term * (1 / delta_t)
    derivatives_pt = torch.where(
        valid_intervalkappa_mask_pt,
        numerator_term_pt / denominator_pt, 
        torch.zeros_like(derivatives_pt) 
    )

    # Optional: Set derivative to 0 outside the original range if needed
    # mask_below_start_pt = (t_query_eff_pt < t_known_pt[0]).unsqueeze(-1)
    # mask_above_end_pt = (t_query_eff_pt > t_known_pt[-1]).unsqueeze(-1)
    # derivatives_pt = torch.where(mask_below_start_pt | mask_above_end_pt, 
    #                              torch.tensor(0.0, device=device, dtype=output_dtype), 
    #                              derivatives_pt)

    return derivatives_pt




def create_spline_interpolator_matrices(
        t_known_pt: torch.Tensor,
        t_potential_queries_pt: torch.Tensor,
        device: torch.device = torch.device('cpu'),
        calc_dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    (v6 离线步骤)
    预先计算所有潜在查询点的插值和导数矩阵。
    """
    if not torchcubicspline_available:
        raise ImportError("torchcubicspline library is not available.")
        
    N = t_known_pt.shape[0]
    G = t_potential_queries_pt.shape[0]

    min_points_for_spline = 2
    if N < min_points_for_spline:
        warnings.warn(f"N 必须 >= {min_points_for_spline} 才能创建样条矩阵。")
        return None

    t_known_float_pt = t_known_pt.to(device=device, dtype=calc_dtype)
    t_potential_queries_float_pt = t_potential_queries_pt.to(device=device, dtype=calc_dtype)
    x_identity = torch.eye(N, device=device, dtype=calc_dtype)

    try:
        coeffs_W = natural_cubic_spline_coeffs(t_known_float_pt, x_identity)
        spline_W = NaturalCubicSpline(coeffs_W)
        W_interp = spline_W.evaluate(t_potential_queries_float_pt)
        W_deriv = spline_W.derivative(t_potential_queries_float_pt)
    except Exception as e:
        warnings.warn(f"创建样条插值器时出错: {e}")
        return None

    return W_interp, W_deriv

import torch
import warnings

# (此处省略 import 和 create_spline_interpolator_matrices 函数)
# 假设 create_spline_interpolator_matrices 保持不变

def batched_tasks_cubic_spline_interpolation_pytorch(
        t_known_pt: torch.Tensor,
        x_known_batched_pt: torch.Tensor,
        t_query_pt: torch.Tensor,         # 支持 (M,) 或 (B, M)
        extrapolate: bool = False,
        t_potential_queries_pt: torch.Tensor | None = None,
        W_interp_potential_pt: torch.Tensor | None = None
    ):
    """
    (v6.3 最终修复版)
    使用 einsum/bmm 优化。
    - 修复了 ndim=2 路径中缺失的 clamp() bug。
    - 移除了函数内部的 sort() 以提高性能。
    - 恢复了最终的 .to(dtype) 转换。
    """
    
    # --- 1. 必需的输入检查 ---
    if W_interp_potential_pt is None or t_potential_queries_pt is None:
        raise ValueError("v6 方案需要 t_potential_queries_pt 和 W_interp_potential_pt。")
    if x_known_batched_pt.ndim != 3:
        raise ValueError("x_known_batched_pt 必须是 3D Tensor (B, N, D)。")
        
    device = x_known_batched_pt.device
    dtype = x_known_batched_pt.dtype # 保存原始 dtype
    B, N, D_val = x_known_batched_pt.shape
    G = t_potential_queries_pt.shape[0]

    # --- 2. 调试断言 (捕获 CUDA 错误) ---
    if torch.isnan(t_query_pt).any():
        raise ValueError("[调试] 错误: t_query_pt 包含 NaN 值！")
    if torch.isinf(t_query_pt).any():
        raise ValueError("[调试] 错误: t_query_pt 包含 Inf 值！")
    if torch.isnan(x_known_batched_pt).any():
        raise ValueError("[调试] 错误: x_known_batched_pt 包含 NaN 值！")
        
    # 检查 t_potential_queries_pt 是否排序 (假设在函数外完成)
    # 使用 .to(torch.float32) 来安全地处理 half 类型
    if not torch.all(t_potential_queries_pt.to(torch.float32).diff() >= 0):
        raise ValueError("[调试] 错误: t_potential_queries_pt 不是单调递增的！请在函数外排序。")

    # 检查 N 维度是否匹配
    if t_known_pt.shape[0] != N:
        raise ValueError(f"[调试] 错误: t_known_pt (N={t_known_pt.shape[0]}) 与 x_known (N={N}) 不匹配。")
    if W_interp_potential_pt.shape[1] != N:
        raise ValueError(f"[调试] 错误: W 矩阵 (N={W_interp_potential_pt.shape[1]}) 与 x_known (N={N}) 不匹配。")
    if W_interp_potential_pt.shape[0] != G:
        raise ValueError(f"[调试] 错误: W 矩阵 (G={W_interp_potential_pt.shape[0]}) 与 t_potential (G={G}) 不匹配。")

    # --- 3. 类型和设备转换 ---
    calc_dtype = torch.promote_types(dtype, torch.float32)
    x_known_float_batched_pt = x_known_batched_pt.to(device=device, dtype=calc_dtype)
    W_interp_float_pt = W_interp_potential_pt.to(device=device, dtype=calc_dtype)
    t_potential_queries_dev = t_potential_queries_pt.to(device)

    # --- 4. 核心逻辑与修复 ---
    
    if t_query_pt.ndim == 1:
        # --- 1D 路径 (M,) ---
        M = t_query_pt.shape[0]
        t_query_dev = t_query_pt.to(device)
        
        indices = torch.searchsorted(t_potential_queries_dev, t_query_dev)
        # 修复 #1: 钳制 (Clamp) (修复由浮点不精确性引起的 G 索引越界)
        indices = torch.clamp(indices, 0, G - 1)
        
        W_truncated = W_interp_float_pt.index_select(0, indices) # (M, N)
        interpolated_x_pt = torch.einsum(
            'mn,bnd->bmd', W_truncated, x_known_float_batched_pt
        )

    elif t_query_pt.ndim == 2:
        # --- 2D 路径 (B, M) ---
        if t_query_pt.shape[0] != B:
            raise ValueError(f"t_query_pt 2D (B={t_query_pt.shape[0]}) 与 x_known (B={B}) 不匹配。")
        M = t_query_pt.shape[1]
        t_query_dev = t_query_pt.to(device) # (B, M)

        indices = torch.searchsorted(t_potential_queries_dev, t_query_dev) # (B, M)
        
        # *** 关键修复！ ***
        # 修复 #1: 钳制 (Clamp) (修复由浮点不精确性引起的 G 索引越界)
        indices = torch.clamp(indices, 0, G - 1) 
        # ******************

        W_truncated = W_interp_float_pt[indices] # (B, M, N)
        interpolated_x_pt = torch.bmm(W_truncated, x_known_float_batched_pt)

    else:
        raise ValueError("t_query_pt 必须是 1D (M,) 或 2D (B,M) Tensor。")
        
    # 修复 #2: 转换回原始 dtype
    return interpolated_x_pt.to(dtype)


def batched_tasks_cubic_spline_interpolation_derivative_pytorch(
        t_known_pt: torch.Tensor,
        x_known_batched_pt: torch.Tensor,
        t_query_pt: torch.Tensor,         # 支持 (M,) 或 (B, M)
        extrapolate: bool = False,
        t_potential_queries_pt: torch.Tensor | None = None,
        W_deriv_potential_pt: torch.Tensor | None = None
    ):
    """
    (v6.3 最终修复版)
    使用 einsum/bmm 优化。
    - 修复了 ndim=2 路径中缺失的 clamp() bug。
    - 移除了函数内部的 sort() 以提高性能。
    - 恢复了最终的 .to(output_dtype) 转换。
    """
    
    # --- 1. 必需的输入检查 ---
    if W_deriv_potential_pt is None or t_potential_queries_pt is None:
        raise ValueError("v6 方案需要 t_potential_queries_pt 和 W_deriv_potential_pt。")
    if x_known_batched_pt.ndim != 3:
        raise ValueError("x_known_batched_pt 必须是 3D Tensor (B, N, D)。")
        
    device = x_known_batched_pt.device
    output_dtype = torch.promote_types(x_known_batched_pt.dtype, torch.float32) # 导数总是 float
    B, N, D_val = x_known_batched_pt.shape
    G = t_potential_queries_pt.shape[0]

    # --- 2. 调试断言 (捕获 CUDA 错误) ---
    if torch.isnan(t_query_pt).any():
        raise ValueError("[调试] 错误: t_query_pt 包含 NaN 值！")
    if torch.isinf(t_query_pt).any():
        raise ValueError("[调试] 错误: t_query_pt 包含 Inf 值！")
    if torch.isnan(x_known_batched_pt).any():
        raise ValueError("[调试] 错误: x_known_batched_pt 包含 NaN 值！")
        
    if not torch.all(t_potential_queries_pt.to(torch.float32).diff() >= 0):
        raise ValueError("[调试] 错误: t_potential_queries_pt 不是单调递增的！请在函数外排序。")

    if t_known_pt.shape[0] != N:
        raise ValueError(f"[调试] 错误: t_known_pt (N={t_known_pt.shape[0]}) 与 x_known (N={N}) 不匹配。")
    if W_deriv_potential_pt.shape[1] != N:
        raise ValueError(f"[调试] 错误: W 矩阵 (N={W_deriv_potential_pt.shape[1]}) 与 x_known (N={N}) 不匹配。")
    if W_deriv_potential_pt.shape[0] != G:
        raise ValueError(f"[调试] 错误: W 矩阵 (G={W_deriv_potential_pt.shape[0]}) 与 t_potential (G={G}) 不匹配。")

    # --- 3. 类型和设备转换 ---
    calc_dtype = torch.promote_types(x_known_batched_pt.dtype, torch.float32)
    x_known_float_batched_pt = x_known_batched_pt.to(device=device, dtype=calc_dtype)
    W_deriv_float_pt = W_deriv_potential_pt.to(device=device, dtype=calc_dtype)
    t_potential_queries_dev = t_potential_queries_pt.to(device)

    # --- 4. 核心逻辑与修复 ---
    
    if t_query_pt.ndim == 1:
        # --- 1D 路径 (M,) ---
        M = t_query_pt.shape[0]
        t_query_dev = t_query_pt.to(device)
        
        indices = torch.searchsorted(t_potential_queries_dev, t_query_dev)
        # 修复 #1: 钳制 (Clamp)
        indices = torch.clamp(indices, 0, G - 1)
        
        W_deriv_truncated = W_deriv_float_pt.index_select(0, indices) # (M, N)
        derivatives_pt = torch.einsum(
            'mn,bnd->bmd', W_deriv_truncated, x_known_float_batched_pt
        )

    elif t_query_pt.ndim == 2:
        # --- 2D 路径 (B, M) ---
        if t_query_pt.shape[0] != B:
            raise ValueError(f"t_query_pt 2D (B={t_query_pt.shape[0]}) 与 x_known (B={B}) 不匹配。")
        M = t_query_pt.shape[1]
        t_query_dev = t_query_pt.to(device) # (B, M)

        indices = torch.searchsorted(t_potential_queries_dev, t_query_dev) # (B, M)
        
        # *** 关键修复！ ***
        # 修复 #1: 钳制 (Clamp)
        indices = torch.clamp(indices, 0, G - 1) 
        # ******************

        W_deriv_truncated = W_deriv_float_pt[indices] # (B, M, N)
        derivatives_pt = torch.bmm(W_deriv_truncated, x_known_float_batched_pt)

    else:
        raise ValueError("t_query_pt 必须是 1D (M,) 或 2D (B,M) Tensor。")
        
    # 修复 #2: 转换回正确的输出 dtype
    return derivatives_pt.to(output_dtype)