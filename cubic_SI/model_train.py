import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

import cubic_SI.computations


class model_trainer(object):
    def __init__(self,lr,n_epochs,batch_size,data_time_list,sample_time_list,gamma,gamma_prime,func_type='None',spline=False,linear=True,decay=0.8,early_stop=False,patience=10,plot_loss=True,save_model=True,
                 save_path='model_history',record_gap=1,conditional=False,conditional_path_encoder=None,hist_times=None,dynamic_conditions=False,data_transformer=None):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.lr = lr
        self.n_epochs = n_epochs 
        self.batch_size = batch_size
        self.gamma = gamma
        self.gamma_prime = gamma_prime
        self.func_type = func_type
        self.spline = spline
        self.linear = linear

        self.decay = decay
        self.early_stop = early_stop
        self.patience = patience
        self.data_time_list = data_time_list
        self.sample_time_list = list(set(sample_time_list)) #list(set(sample_time_list).difference(set(data_time_list)))
        
        self.loss_history={'loss':[],"loss_b":[],"loss_k":[]}
        self.plot_loss = plot_loss
        self.save_model = save_model
        self.save_path = save_path
        self.record_gap = record_gap
        
        self.conditional = conditional
        self.t_potential = torch.sort(torch.Tensor(self.sample_time_list).to(self.device))[0]

        self.W_interp, self.W_deriv = cubic_SI.computations.create_spline_interpolator_matrices(
            torch.Tensor(self.data_time_list).to(self.device), self.t_potential, self.device
        )
        
        self.conditional_path_encoder = conditional_path_encoder
        self.hist_times = hist_times
        self.dynamic_conditions=dynamic_conditions
        self.data_transformer = data_transformer
        if self.data_transformer is not None:
            self.data_transformer = self.data_transformer.to(self.device)



    # ==============================================================================
    # Helper Function to Count Parameters
    # ==============================================================================
    def count_parameters(self, model):
        """Counts the total number of trainable parameters in a PyTorch model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def sample_without_neighborhood(self, sample_time_list, data_time_list, batch_size, epsilon=0.01):
        """
        在sample_time_list范围内采样，避开data_time_list中所有点的epsilon邻域
        
        参数:
            sample_time_list: 原始采样时间范围列表
            data_time_list: 需要避开的时间点列表
            batch_size: 采样数量
            epsilon: 邻域半径
        返回:
            符合条件的采样结果
        """
        # 确定原始采样范围
        min_t = min(sample_time_list)
        max_t = max(sample_time_list)
        
        # 生成所有需要排除的区间 [t-epsilon, t+epsilon]
        forbidden_intervals = []
        for t in data_time_list:
            start = max(min_t, t - epsilon)  # 确保不超出原始范围
            end = min(max_t, t + epsilon)
            forbidden_intervals.append((start, end))
        
        # 合并重叠的禁止区间（避免重复排除）
        if not forbidden_intervals:
            merged = []
        else:
            # 按区间起点排序
            sorted_intervals = sorted(forbidden_intervals, key=lambda x: x[0])
            merged = [sorted_intervals[0]]
            for current in sorted_intervals[1:]:
                last = merged[-1]
                if current[0] <= last[1]:
                    # 重叠区间，合并它们
                    merged[-1] = (last[0], max(last[1], current[1]))
                else:
                    merged.append(current)
        
        # 计算有效采样区间（原始范围减去禁止区间）
        valid_intervals = []
        prev_end = min_t
        for (start, end) in merged:
            if prev_end < start:
                valid_intervals.append((prev_end, start))
            prev_end = max(prev_end, end)
        if prev_end < max_t:
            valid_intervals.append((prev_end, max_t))
        
        # 检查是否有有效区间
        if not valid_intervals:
            raise ValueError("所有可能的采样区间都被禁止邻域覆盖，请调整epsilon或采样范围")
        
        # 按区间长度分配采样数量（保证每个区间的采样概率与其长度成正比）
        interval_lengths = [end - start for (start, end) in valid_intervals]
        total_length = sum(interval_lengths)
        sample_counts = []
        remaining = batch_size
        
        # 为每个区间分配采样数量
        for i, length in enumerate(interval_lengths[:-1]):
            proportion = length / total_length
            count = int(round(proportion * batch_size))
            count = max(0, min(count, remaining))
            sample_counts.append(count)
            remaining -= count
        sample_counts.append(remaining)  # 最后一个区间分配剩余数量
        
        # 在每个有效区间内采样
        samples = []
        for i, (start, end) in enumerate(valid_intervals):
            count = sample_counts[i]
            if count > 0:
                # 在[start, end]区间内均匀采样
                samples.extend(np.random.uniform(start, end, count))
        
        # 打乱采样结果顺序
        np.random.shuffle(samples)
        return np.array(samples)
    
    
    def loss_plot(self):
        list1 = self.loss_history['loss']
        x = range(len(list1))
        plt.plot(x, list1, label='loss')
        plt.legend()
        plt.show()
        
    
    def model_save(self,b_m,kappa_m):
        models_dict = {
            'b_m': b_m.state_dict(),
            'kappa_m': kappa_m.state_dict(),
            'loss_history':self.loss_history
                    }
        if self.conditional_path_encoder is not None:
            models_dict['conditional_path_encoder'] = self.conditional_path_encoder.state_dict()
        torch.save(models_dict, self.save_path+'/model.pt')
        

    def train_an_epoch(self,b_m,kappa_m,optimizer,dataloader,b_params=None,s_params=None):
        epoch_loss = 0
        epoch_loss_b = 0
        epoch_loss_k = 0
        data_time_list = torch.Tensor(self.data_time_list).to(self.device)
        for data in dataloader:
            conditions = None
            if self.conditional:
                conditions = data[1]
                data = data[0]
                conditions = conditions.to(self.device)
            data = data.to(self.device)
            if self.data_transformer is not None:
                data = self.data_transformer(data)
            batch_size = data.shape[0]
            
            t = np.random.choice(self.sample_time_list,batch_size)
            # t = np.random.uniform(self.sample_time_list[0],self.sample_time_list[-1],batch_size)
            # t = self.sample_without_neighborhood(
            #     self.sample_time_list, 
            #     self.data_time_list, 
            #     batch_size, 
            #     epsilon=0.01
            # )
            t = torch.from_numpy(t).reshape(-1,1).to(self.device)
            if self.dynamic_conditions:
                obs_times = torch.tensor(self.data_time_list, device=self.device).unsqueeze(0)
                conditions = conditions[torch.arange(batch_size), torch.argmin(torch.abs(t - obs_times), dim=1)]
            if self.linear:
                x_t=cubic_SI.computations.batched_tasks_custom_linear_interpolation_pytorch(data_time_list,data,t).squeeze()
            elif self.spline:
                x_t=cubic_SI.computations.batched_tasks_cubic_spline_interpolation_pytorch(data_time_list,data,t,
                t_potential_queries_pt=self.t_potential,
                W_interp_potential_pt=self.W_interp).squeeze()
            else:
                return

            z = torch.randn(x_t.shape).to(self.device)
            x_t = x_t + z * self.gamma(t, data_time_list, self.func_type)
            

            if self.conditional:
                if self.conditional_path_encoder is not None:
                    conditions = self.conditional_path_encoder(conditions.float(),torch.tensor(self.hist_times,device=self.device).repeat(conditions.shape[0],1).float())
                bias=b_m(x_t.float(),t.float(),conditions.float())
                kappa=kappa_m(x_t.float(),t.float(),conditions.float())
            else:
                bias=b_m(x_t.float(),t.float())
                kappa=kappa_m(x_t.float(),t.float())
            
            
            if self.linear:
                target = self.gamma_prime(t, data_time_list, self.func_type) * z + \
                            cubic_SI.computations.batched_tasks_custom_linear_interpolation_derivative_pytorch(data_time_list, data, t).squeeze()
                bias_loss = bias_loss = F.mse_loss(bias.float(), target.float())
                # bias_loss = torch.mean(0.5 * (bias**2).sum(dim=list(range(1, bias.dim()))) - ((self.gamma_prime(t, data_time_list, self.func_type) * z + cubic_SI.computations.batched_tasks_custom_linear_interpolation_derivative_pytorch(data_time_list,data,t).squeeze()) * bias).sum(dim=list(range(1, bias.dim()))))
                
                
            elif self.spline:
                # bias_loss = torch.mean(0.5 * (bias**2).sum(dim=list(range(1, bias.dim()))) - ((self.gamma(t, data_time_list, self.func_type) * z + cubic_SI.computations.batched_tasks_cubic_spline_interpolation_derivative_pytorch(data_time_list,data,t).squeeze()) * bias).sum(dim=list(range(1, bias.dim()))))
                target = self.gamma_prime(t, data_time_list, self.func_type) * z + \
                            cubic_SI.computations.batched_tasks_cubic_spline_interpolation_derivative_pytorch(data_time_list, data, t, t_potential_queries_pt=self.t_potential, W_deriv_potential_pt=self.W_deriv).squeeze()

                bias_loss = bias_loss = F.mse_loss(bias.float(), target.float())
                

            if self.func_type == 'None':
                kappa_loss = torch.tensor(0.0).to(self.device)
            else:
                # score_loss = torch.mean(0.5 * (score**2).sum(dim=list(range(1, score.dim()))) + ((self.gamma(t, data_time_list, self.func_type))**(-1) * (self.gamma(t, data_time_list, self.func_type))**(-1) * z) * score).sum(dim=list(range(1, score.dim()))))
                # score_loss = F.mse_loss(score.float(), torch.clamp(-(self.gamma(t, data_time_list, self.func_type))**(-1) * z,max=1e4,min=-1e4)).float()
                kappa_loss = F.mse_loss(kappa.float(),-z.float())
            loss=bias_loss + kappa_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b_m.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(kappa_m.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss
            epoch_loss_b += bias_loss
            epoch_loss_k += kappa_loss
        return epoch_loss, epoch_loss_b, epoch_loss_k


    def train(self,b_m,kappa_m,dataloader):
        
        print(self.count_parameters(b_m))
        print(self.count_parameters(kappa_m))
        
        optimizer = optim.Adam(list(b_m.parameters())+list(kappa_m.parameters()), lr=self.lr)
        if self.conditional_path_encoder is not None:
            optimizer = optim.Adam(list(b_m.parameters())+list(kappa_m.parameters())+list(self.conditional_path_encoder.parameters()), lr=self.lr)

        if self.early_stop:
            best_loss = np.inf
            epochs_no_improve = 0
            stop_indicator = False
        
        with tqdm(total=self.n_epochs, mininterval=1.0) as pbar:
            for n in range(self.n_epochs):
                if self.early_stop:
                    if stop_indicator:
                        print("Early stopping at epoch", n)
                        break

                epoch_loss, epoch_loss_b, epoch_loss_k = self.train_an_epoch(b_m,kappa_m,optimizer,dataloader)

                
                if (n+1) % self.record_gap == 0: 
                    with torch.no_grad():
                        self.loss_history['loss'].append(epoch_loss.item())
                        self.loss_history['loss_b'].append(epoch_loss_b.item())
                        self.loss_history['loss_k'].append(epoch_loss_k.item())
                pbar.set_description('processed: %d' % (1 + n))
                pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy(),'loss_b':epoch_loss_b.detach().cpu().numpy(),'loss_k':epoch_loss_k.detach().cpu().numpy(),})
                pbar.update(1)
                
                if self.early_stop:                
                    if epoch_loss.item() < best_loss:
                        best_loss = epoch_loss.item()
                        epochs_no_improve = 0  
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        stop_indicator = True
                        
        if self.save_model:
            self.model_save(b_m,kappa_m)
            torch.cuda.empty_cache()
        if self.plot_loss:
            self.loss_plot()

