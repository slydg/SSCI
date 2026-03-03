import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cubic_SI.utils
import math


class ResBlock(nn.Module):
    
    def __init__(self,hidden_dimension,t_size):
        super(ResBlock, self).__init__()

        self.fc1 = nn.Linear(hidden_dimension,2*hidden_dimension)
        self.relu = nn.LeakyReLU()
        self.adaLN = StyleAdaptiveLayerNorm(2*hidden_dimension,t_size)
        self.fc2 = nn.Linear(2*hidden_dimension,hidden_dimension) 

    def forward(self,x,y):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.adaLN(out,y)
        out = self.fc2(out)
        return out + x



class UNetWithLinear(nn.Module):
    def __init__(self, x_size, t_size, output_size, hidden_size=64, n_layers=4, condition_input_size=None,bounded_out=False,concentrate=None):
        super(UNetWithLinear, self).__init__()

        self.time_embedder = TimeEmbedding(output_size=t_size)
        
        # MODIFICATION: Conditionally create the condition embedder
        self.condition_embedder = None
        if condition_input_size is not None:
            self.condition_embedder = ConditionEncoder(output_size=t_size, input_size=condition_input_size)

        self.concentrate=concentrate
        if concentrate is not None:
            self.encoder = nn.Sequential(
                nn.Linear(x_size, concentrate), nn.ReLU(),
                nn.Linear(concentrate, hidden_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, concentrate), nn.ReLU(),
                nn.Linear(concentrate, output_size)
            )
        else:
            self.encoder = nn.Linear(x_size, hidden_size)
            self.decoder = nn.Linear(hidden_size, output_size)
            
        self.trans_lst = nn.ModuleList([ResBlock(hidden_size, t_size) for _ in range(n_layers)])
        self.bounded_out=bounded_out

    def forward(self, x_t, t, conditions=None):
        """
        The forward pass now handles an optional conditions tensor.

        Args:
            x_t (Tensor): The input data. Shape: [B, x_size]
            t (Tensor): The timestep. Shape: [B, 1]
            conditions (Tensor, optional): The physical parameters. Shape: [B, 5]. Defaults to None.
        """
        # 1. Always generate the time embedding
        time_embedding = self.time_embedder(t)
        
        # Initialize the combined embedding with the time embedding
        combined_embedding = time_embedding

        # MODIFICATION: Conditionally add the physical condition embedding
        if conditions is not None:
            # Safety check: ensure the model was initialized to be conditional
            if self.condition_embedder is None:
                raise ValueError("Model was initialized without a condition_embedder, but conditions were provided.")
            
            # 2. Generate and add the physical condition embedding
            condition_embedding = self.condition_embedder(conditions)
            combined_embedding = combined_embedding + condition_embedding

        # The rest of the network proceeds as before, using the final combined_embedding
        x_t = self.encoder(x_t)

        for trans in self.trans_lst:
            x_t = trans(x_t, combined_embedding)

        x_t = self.decoder(x_t)
        return x_t





class TimeEmbedding(nn.Module):
    def __init__(self,output_size,hidden_size=16,input_size=1):
        super(TimeEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    def forward(self,t):
        t=F.leaky_relu(self.fc1(t))
        t=self.fc2(t)
        return t


class ConditionEncoder(nn.Module):
    def __init__(self, output_size, hidden_size=32, input_size=5):
        super(ConditionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, conditions):
        embedding = F.leaky_relu(self.fc1(conditions))
        embedding = self.fc2(embedding)
        return embedding



    
class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[:self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels:], 0)

    def forward(self, x, c):
        """
        Parameters:
        x (Tensor): The input feature maps with shape [batch_size, time, in_channels].
        c (Tensor): The conditioning input with shape [batch_size, 1, cond_channels].
        
        Returns:
        Tensor: The modulated feature maps with the same shape as input x.
        """
        saln_params = self.saln(c)
        gamma, beta = torch.chunk(saln_params, chunks=2, dim=-1)
        
        out = self.norm(x)
        out = gamma * out + beta
        
        return out
    
    
    
    
class MLPNet_Fair(nn.Module):
    """
    内部结构被设计为与 L-ODE/SDE 基线 (Encoder + DynamicsNet + Decoder)
    的参数量和结构相匹配, 以实现公平比较。
    
    <<< V2 UPDATE : >>>
    project_to_latent (A.2) 现在也接收 'conditions',
    以完全匹配 L-ODE/SDE 的 z0_head 结构。
    """
    def __init__(self, 
                 data_dim,              # 原始数据维度
                 cond_dim,              # 原始条件维度
                 latent_dim,            # L-ODE/SDE 使用的潜维度
                 vae_hidden_dim,        # L-ODE/SDE 编码器/解码器使用的隐藏维度
                 dyn_hidden_dim,        # L-ODE/SDE 动力学网络使用的隐藏维度
                 **kwargs):             # 接受多余的 config 参数
        super().__init__()

        # --- (A) VAE 编码器 (的 "副本") ---
        # (与 L-ODE/SDE.encoder_mlp 结构相同)
        # 作用: x_t -> h (数据空间 -> VAE隐藏空间)
        self.vae_encoder_mlp = nn.Sequential(
            nn.Linear(data_dim, vae_hidden_dim), nn.ReLU(),
            nn.Linear(vae_hidden_dim, vae_hidden_dim)
        )
        
        # --- (A.2) 投影层 (的 "副本") ---
        # (模仿 L-ODE/SDE.z0_head)
        # <<< MODIFIED: 输入现在是 (vae_hidden_dim + cond_dim) >>>
        # 作用: (h + c) -> z (VAE隐藏空间 + 条件 -> 潜空间)
        self.project_to_latent = nn.Linear(vae_hidden_dim + cond_dim, latent_dim)

        # --- (B) 动力学网络 (的 "副本") ---
        # (与 L-ODE/SDE 的 _LatentODEFunc.net 结构相同)
        # 作用: z + t + c -> z' (在潜空间中处理动力学)
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + 1 + cond_dim, dyn_hidden_dim), nn.Tanh(),
            nn.Linear(dyn_hidden_dim, dyn_hidden_dim), nn.Tanh(),
            nn.Linear(dyn_hidden_dim, latent_dim)
        )
        
        # --- (C) VAE 解码器 (的 "副本") ---
        # (与 L-ODE/SDE.decoder 结构相同)
        # 作用: z' -> epsilon (潜空间 -> 数据空间)
        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_dim, vae_hidden_dim), nn.ReLU(),
            nn.Linear(vae_hidden_dim, vae_hidden_dim), nn.ReLU(),
            nn.Linear(vae_hidden_dim, data_dim)
        )

    def forward(self, x_t, t, conditions):
        """
        Args:
            x_t (Tensor): 带噪数据。Shape: [B, data_dim]
            t (Tensor): 时间步。Shape: [B, 1] (或标量)
            conditions (Tensor): 物理参数。Shape: [B, cond_dim]
        
        Returns:
            epsilon_pred (Tensor): 预测的噪声。Shape: [B, data_dim]
        """
        
        # 确保 t 是 [B, 1]
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] != x_t.shape[0]):
             t_batch = t.expand(x_t.shape[0], 1).to(x_t.dtype)
        elif t.dim() == 1:
             t_batch = t.unsqueeze(-1)
        else:
             t_batch = t
            
        # 1. 编码 (x_t -> h)
        h = self.vae_encoder_mlp(x_t)
        
        # 2. 投影 ((h + c) -> z)
        # <<< MODIFIED: 在投影前注入条件 >>>
        h_with_cond = torch.cat([h, conditions], dim=1)
        z = self.project_to_latent(h_with_cond)
        
        # 3. 运行 "动力学" (z + t + c -> z')
        #    (在潜空间中再次注入 t 和 c)
        dyn_input = torch.cat([z, t_batch, conditions], dim=1)
        z_prime = self.dynamics_net(dyn_input)
        
        # 4. 解码 (z' -> epsilon_pred)
        epsilon_pred = self.vae_decoder(z_prime)
        
        return epsilon_pred
    
    
class MotionTransformerEncoder(nn.Module):
    def __init__(self, input_dim=135, embed_dim=256, num_heads=4, layers=4, latent_dim=256):
        super().__init__()
        
        # 1. 特征投影
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # 2. 时间编码 (Sinusoidal)
        # 也可以用 nn.Embedding 学习，但 Sinusoidal 对连续时间泛化更好
        self.embed_dim = embed_dim
        
        # 3. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # 4. 输出投影 (将序列压缩为单个向量 c)
        # 这里使用简单的 Attention Pooling 或取 Mean
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        

    def get_sinusoidal_encoding(self, times):
        # times: (B, T)
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=times.device) * -emb)
        emb = times.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb # (B, T, embed_dim)

    def forward(self, x, times):
        """
        x: (B, 12, 135) - 稀疏动作历史
        times: (B, 12)  - 对应的负数时间点
        """
        B, T, _ = x.shape
        
        # Embedding
        x_embed = self.input_proj(x) # (B, T, E)
        
        # Add Time Encoding
        t_embed = self.get_sinusoidal_encoding(times)
        x_embed = x_embed + t_embed
        
        # Transformer Pass
        hidden = self.transformer(x_embed) # (B, T, E)
        
        # Pooling: 取平均，或者取最后一帧 (建议取平均以获得全局概况)
        c = hidden.mean(dim=1) # (B, E)
        
        return self.output_proj(c) # (B, latent_dim)