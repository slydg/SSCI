import math
import torch
import glob
import os
import random
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.pt")))
        self.shuffle = shuffle
        
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def __iter__(self):
        # --- 优化点 1: 获取当前 Worker 信息 ---
        worker_info = get_worker_info()
        
        # 复制文件列表以避免修改全局状态
        files_to_read = list(self.files)

        if worker_info is not None:
            # 如果是多进程模式，将文件列表均分给每个 Worker
            per_worker = int(math.ceil(len(files_to_read) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(files_to_read))
            
            # 当前 Worker 只处理这一部分文件
            files_to_read = files_to_read[iter_start:iter_end]

        # --- 优化点 2: 仅 Shuffle 分配到的文件 ---
        if self.shuffle:
            random.shuffle(files_to_read)
            
        for f_path in files_to_read:
            try:
                # --- 优化点 3: 明确 map_location 减少 GPU 上下文开销 (如果只在 CPU 做预处理) ---
                data_dict = torch.load(f_path, map_location='cpu')
                
                trajs = data_dict['target_envelopes'] 
                conds = data_dict['conditions']       
                
                num_samples = trajs.shape[0]
                indices = torch.randperm(num_samples).tolist() if self.shuffle else range(num_samples)
                
                # --- 优化点 4: 直接 Yield 避免额外的 Python 循环开销 ---
                # 这种写法在 Python 层面比手动 for i in indices 更快一点
                for i in indices:
                    yield trajs[i], conds[i]
                    
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load {f_path}. Error: {e}")
                continue
            
class MmapDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir):
        meta = np.load(os.path.join(processed_dir, 'meta.npy'), allow_pickle=True).item()
        self.length = meta['samples']
        
        # mmap_mode='r' 是关键！它不会把数据读入内存，而是建立了映射
        self.trajs = np.memmap(
            os.path.join(processed_dir, 'trajs.npy'), 
            dtype='float32', 
            mode='r', 
            shape=(self.length, *meta['traj_shape'])
        )
        self.conds = np.memmap(
            os.path.join(processed_dir, 'conds.npy'), 
            dtype='float32', 
            mode='r', 
            shape=(self.length, *meta['cond_shape'])
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 这一步非常快，操作系统会自动处理缓存
        # 必须转为 copy 或者是 torch.from_numpy，否则可能会有负 stride 问题
        t = torch.from_numpy(np.array(self.trajs[idx]))
        c = torch.from_numpy(np.array(self.conds[idx]))
        return t, c