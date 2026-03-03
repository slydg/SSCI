import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import torch
import torch.nn.functional as F
import shutil
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset, IterableDataset
from braindecode.datasets import HGD
from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
import gc
import glob
import random
import pandas as pd
import sklearn
from scipy import interpolate
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML, display
import mne

# Assuming cubic_SI is a local module available in your environment
import cubic_SI
from cubic_SI import model
from cubic_SI import computations as cmp

# ==============================================================================
# 0. Utils & Setup
# ==============================================================================
def setup_seed(seed: int = 42, deterministic: bool = False):
    """
    Sets random seeds for PyTorch, NumPy, and Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

def restore_current_state(data, n_orig_ch):
    """Restores tensor to numpy and crops extra channels if necessary."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.shape[-1] == n_orig_ch:
        return data
    if data.shape[-1] > n_orig_ch:
        return data[..., :n_orig_ch]
    return data

def to_mne_format(data):
    """Ensures format is (N, C, T) for MNE functions."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.ndim >= 2:
        # If shape is (..., T, C), swap to (..., C, T)
        if data.shape[-1] < data.shape[-2]: 
            data = np.swapaxes(data, -1, -2)
    return data

def get_global_scale(real, models_dict):
    """Calculates global vmin/vmax for consistent colorbars."""
    all_data = [real.flatten()]
    for v in models_dict.values():
        all_data.append(v.flatten())
    combined = np.concatenate(all_data)
    limit = np.percentile(np.abs(combined), 99)
    return -limit, limit

def resample_batch_to_target(batch_data, target_length):
    """Resamples data to match the target time length."""
    N, T_curr, C = batch_data.shape
    if T_curr == target_length: return batch_data
    
    intervals_curr = T_curr - 1
    intervals_target = target_length - 1
    if intervals_target == 0: return batch_data[:, 0:1, :]
    
    stride = int(intervals_curr / intervals_target)
    resampled = batch_data[:, ::stride, :]
    
    if resampled.shape[1] != target_length:
        resampled = resampled[:, :target_length, :]
    return resampled

# ==============================================================================
# 1. DSP Tools (Signal Decomposition)
# ==============================================================================
def get_sos_bank(fs, f_start=4, f_end=38):
    """Generates a bank of Butterworth bandpass filters."""
    freqs = np.arange(f_start, f_end + 1)
    sos_bank = []
    nyq = 0.5 * fs
    half_width = 0.5 
    for f in freqs:
        low = max(0.01, (f - half_width) / nyq)
        high = min(0.99, (f + half_width) / nyq)
        sos = signal.butter(2, [low, high], btype='band', output='sos')
        sos_bank.append(sos)
    return sos_bank, freqs

def decompose_acausal(X_batch, sos_bank):
    """
    Zero-phase filtering (filtfilt) + Hilbert Transform.
    Input: (B, T, C) -> Output: Envelopes, Phases (B, T, C, F)
    """
    B, T, C = X_batch.shape
    n_freqs = len(sos_bank)
    amp_out = np.zeros((B, T, C, n_freqs), dtype=np.float32)
    pha_out = np.zeros((B, T, C, n_freqs), dtype=np.float32)
    
    # Flatten time for filtering: (T, B*C)
    X_reshaped = X_batch.transpose(1, 0, 2).reshape(T, -1) 
    
    for i, sos in enumerate(sos_bank):
        # sosfiltfilt = Zero Phase (Forward + Backward)
        filtered = signal.sosfiltfilt(sos, X_reshaped, axis=0)
        analytic = signal.hilbert(filtered, axis=0)
        
        amp_out[:, :, :, i] = np.abs(analytic).reshape(T, B, C).transpose(1, 0, 2)
        pha_out[:, :, :, i] = np.angle(analytic).reshape(T, B, C).transpose(1, 0, 2)
        
    return amp_out, pha_out

# ==============================================================================
# 2. Data Loader & Processing
# ==============================================================================
def load_hgd_dataset_object(subject_ids, target_fs, target_len_samples, buffer_sec=0.0, is_train=True):
    print(f"\n[Loader] Initializing HGD (Train={is_train})...")
    dataset = HGD(subject_ids=subject_ids)
    
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False), 
        Preprocessor('resample', sfreq=target_fs), 
        Preprocessor('filter', l_freq=4., h_freq=38.),           
        Preprocessor(fn=lambda x: x * 1e6, apply_on_array=True),  
        Preprocessor('set_eeg_reference', ref_channels='average'), 
    ]
    preprocess(dataset, preprocessors)
    
    info = dataset.datasets[0].raw.info
    sfreq = info['sfreq']
    buffer_samples = int(buffer_sec * sfreq)
    
    full_window_size = target_len_samples + (2 * buffer_samples)
    
    start_offset = -buffer_samples
    stride = int(target_len_samples * 0.2) if is_train else target_len_samples 
    mapping = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'rest': 3}
    
    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=start_offset,
        trial_stop_offset_samples=0, window_size_samples=full_window_size,
        window_stride_samples=stride, preload=True, mapping=mapping
    )
    return windows_dataset, info, sfreq, buffer_samples

def verify_train_reconstruction(dataset, fs, buffer_samples, save_dir="vis_check"):
    """Check Train Set reconstruction quality."""
    print(f"\n[Check] Verifying Train Reconstruction...")
    os.makedirs(save_dir, exist_ok=True)
    
    X_raw = dataset[0][0]
    X_batch = torch.tensor(X_raw[None, ...]).permute(0, 2, 1).numpy()
    sos_bank, _ = get_sos_bank(fs, 4, 38)
    amp, pha = decompose_acausal(X_batch, sos_bank)
    rec_sig = np.sum(amp * np.cos(pha), axis=-1)
    
    start, end = buffer_samples, X_batch.shape[1] - buffer_samples
    r2 = r2_score(X_batch[0, start:end, 0], rec_sig[0, start:end, 0])
    
    plt.figure(figsize=(10, 5))
    plt.plot(X_batch[0, :, 0], 'k', alpha=0.3, label='Original')
    plt.plot(np.arange(start, end), rec_sig[0, start:end, 0], 'r--', label='Reconstructed Core')
    plt.axvline(start, color='b', linestyle=':'); plt.axvline(end, color='b', linestyle=':')
    plt.title(f"Train Recon (R2={r2:.3f})")
    plt.savefig(os.path.join(save_dir, "verify_train.png"))
    plt.close()
    print(f"   Train R2: {r2:.4f}")

def verify_val_extraction(X_full_low, amp_sym, global_center_idx, local_extract_idx, save_dir="vis_check"):
    """Verify extracted point from Symmetric Window visually."""
    print(f"\n[Check] Verifying Val Symmetric Extraction...")
    os.makedirs(save_dir, exist_ok=True)
    
    ch_idx, freq_idx = 0, 10
    
    # 1. Calculate Full Oracle for comparison (Reference)
    sos_bank, _ = get_sos_bank(100.0, 4, 38) # Assume 100Hz Low FS
    amp_full, _ = decompose_acausal(X_full_low, sos_bank)
    env_full = amp_full[0, :, ch_idx, freq_idx]
    
    # 2. Get Symmetric Window Envelope
    env_sym = amp_sym[0, :, ch_idx, freq_idx]
    
    # 3. Create Time Axes for alignment
    t_full = np.arange(len(env_full))
    start_offset = global_center_idx - local_extract_idx
    t_sym = np.arange(len(env_sym)) + start_offset
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_full, env_full, 'k-', alpha=0.3, linewidth=3, label='Full Sequence Reference')
    plt.plot(t_sym, env_sym, 'b--', label='Symmetric Window (Used)')
    
    val_point = env_sym[local_extract_idx]
    plt.scatter([global_center_idx], [val_point], c='r', s=150, marker='x', zorder=5, label='Extracted Start (t=0)')
    
    plt.axvline(x=global_center_idx, color='r', linestyle=':', alpha=0.5)
    plt.title(f"Val Check: Symmetric Window Alignment (Ch{ch_idx})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "verify_val_symmetric.png"))
    plt.close()
    print(f"   [Plot] Saved verification to {save_dir}")

def process_train_chunks(dataset, output_dir, batch_size, sfreq, buffer_samples):
    """Train: Use standard buffered window -> Decompose -> Crop -> Save."""
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    
    sos_bank, _ = get_sos_bank(sfreq, 4, 38)
    chunk_idx = 0
    
    print(f"\n[Processor] Processing Train Chunks...")
    
    for batch_X, batch_y, _ in loader:
        X_np = batch_X.numpy().transpose(0, 2, 1).astype(np.float32) # (B, T, C)
        y_onehot = np.eye(4)[batch_y.numpy()].astype(np.float32)
        
        # 1. Decompose full window
        amp_full, _ = decompose_acausal(X_np, sos_bank)
        
        # 2. Crop Buffer
        if buffer_samples > 0:
            amp_core = amp_full[:, buffer_samples:-buffer_samples, :, :]
        else:
            amp_core = amp_full
            
        B, T, C, F = amp_core.shape
        env_flat = amp_core.reshape(B, T, -1) 
        
        torch.save({
            'target_envelopes': torch.tensor(env_flat, dtype=torch.float32), 
            'conditions': torch.tensor(y_onehot, dtype=torch.float32)
        }, os.path.join(output_dir, f"chunk_{chunk_idx:04d}.pt"))
        chunk_idx += 1
        
    print(f"[Processor] Saved {chunk_idx} training chunks.")
    del loader, sos_bank
    gc.collect()

def process_val_symmetric_oracle(dataset, output_dir, sfreq_high, sfreq_low, history_sec, batch_size=32, plot_dir=None):
    """Val (Symmetric Oracle): Chunked processing."""
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    sos_bank, _ = get_sos_bank(sfreq_low, 4, 38)
    chunk_idx = 0
    has_verified = False
    
    print(f"\n[Processor] Processing Val Chunks (Symmetric Oracle)...")
    
    for batch_X, batch_y, _ in loader:
        X_high = batch_X.numpy().transpose(0, 2, 1).astype(np.float32)
        y_onehot = np.eye(4)[batch_y.numpy()].astype(np.float32)
        
        B, T_high, C = X_high.shape
        split_idx_high = int(history_sec * sfreq_high)
        X_future_high = X_high[:, split_idx_high:, :]
        
        num_samples_low = int(T_high * (sfreq_low / sfreq_high))
        X_low = signal.resample(X_high, num_samples_low, axis=1)
        
        idx_center = int(history_sec * sfreq_low)
        half_window_pts = int(1.0 * sfreq_low) 
        
        idx_start = idx_center - half_window_pts
        idx_end = idx_center + half_window_pts
        
        if idx_start < 0: idx_start = 0
        if idx_end > X_low.shape[1]: idx_end = X_low.shape[1]
            
        X_symmetric = X_low[:, idx_start:idx_end, :] 
        amp_sym, pha_sym = decompose_acausal(X_symmetric, sos_bank)
        
        extract_idx = idx_center - idx_start
        
        if not has_verified and plot_dir is not None:
             print(f"[Val] Verifying batch 0 symmetric extraction...")
             verify_val_extraction(X_low, amp_sym, idx_center, extract_idx, save_dir=plot_dir)
             has_verified = True

        start_env = amp_sym[:, extract_idx, :, :] 
        start_pha = pha_sym[:, extract_idx, :, :] 
        
        B_out, C_out, F_out = start_env.shape
        torch.save({
            'conditions': torch.tensor(y_onehot, dtype=torch.float32),
            'start_envelope': torch.tensor(start_env.reshape(B_out, -1), dtype=torch.float32),
            'start_phase': torch.tensor(start_pha.reshape(B_out, -1), dtype=torch.float32),
            'gt_raw_voltage': torch.tensor(X_future_high, dtype=torch.float32),
            'n_freqs': F_out
        }, os.path.join(output_dir, f"val_chunk_{chunk_idx:04d}.pt"))
        
        chunk_idx += 1
        del X_high, X_low, X_symmetric, amp_sym, pha_sym
    
    print(f"[Val] Saved {chunk_idx} validation chunks to {output_dir}")
    del loader, sos_bank
    gc.collect()

def run_train_phase(root_dir, subjects, fs, len_target, buffer_sec, buffer_samples, history_sec, fs_val):
    """Encapsulates the training data generation process."""
    print("="*40 + "\n STARTING TRAIN PHASE \n" + "="*40)
    train_ds, info, sfreq_train, buf_pts_train = load_hgd_dataset_object(
        subjects, fs, len_target, buffer_sec, is_train=True
    )
    verify_train_reconstruction(train_ds, sfreq_train, buf_pts_train, save_dir=os.path.join(root_dir, "vis"))
    process_train_chunks(
        train_ds, os.path.join(root_dir, "train"), 
        batch_size=512, sfreq=sfreq_train, buffer_samples=buf_pts_train
    )
    sos_bank, _ = get_sos_bank(fs, 4, 38)
    meta = {
        'info': info, 'sfreq_train': fs, 'sfreq_val': fs_val,        
        'n_channels': train_ds[0][0].shape[0], 'n_freqs': len(sos_bank),
        'history_sec': history_sec
    }
    torch.save(meta, os.path.join(root_dir, "dataset_meta.pt"))
    print("[Train Phase] Cleaning up memory...")
    del train_ds, info, sos_bank
    gc.collect()

def run_val_phase(root_dir, subjects, fs_val, fs_train, len_total, history_sec):
    """Encapsulates the validation data generation process."""
    print("="*40 + "\n STARTING VAL PHASE \n" + "="*40)
    val_ds, _, _, _ = load_hgd_dataset_object(
        subjects, fs_val, len_total, buffer_sec=0.0, is_train=False
    )
    process_val_symmetric_oracle(
        val_ds, 
        output_dir=os.path.join(root_dir, "val"),
        sfreq_high=fs_val, 
        sfreq_low=fs_train, 
        history_sec=history_sec, 
        batch_size=32,
        plot_dir=os.path.join(root_dir, "vis")
    )
    print("[Val Phase] Cleaning up memory...")
    del val_ds
    gc.collect()

def get_train_time_vector(root_out_dir):
    """Reconstructs the time vector 't' for the training set."""
    meta_path = os.path.join(root_out_dir, "dataset_meta.pt")
    train_dir = os.path.join(root_out_dir, "train")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    meta = torch.load(meta_path, weights_only=False)
    sfreq = meta['sfreq_train']
    
    search_path = os.path.join(train_dir, "chunk_*.pt")
    files = sorted(glob.glob(search_path))
    if not files: raise FileNotFoundError(f"No chunk files found in {train_dir}")
        
    print(f"[Time Loader] Inspecting {files[0]} for sequence length...")
    first_chunk = torch.load(files[0])
    
    if 'target_envelopes' not in first_chunk:
        raise KeyError(f"Key 'target_envelopes' not found. Found: {first_chunk.keys()}")

    T = first_chunk['target_envelopes'].shape[1]
    duration = (T - 1) / sfreq
    t_np = np.linspace(0, duration, T)
    t_tensor = torch.tensor(t_np, dtype=torch.float32)
    
    print(f"[Time Loader] Reconstructed t. Shape: {t_tensor.shape}")
    print(f"            Fs: {sfreq}Hz, Duration: {duration:.2f}s")
    return t_tensor

# ==============================================================================
# 3. Dataset & DataLoader
# ==============================================================================
class FileLevelDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.pt")))
        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {data_dir}")
        print(f"Found {len(self.files)} chunk files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_path = self.files[idx]
        data = torch.load(f_path, map_location='cpu')
        return data['target_envelopes'], data['conditions']

def raw_collate_fn(batch):
    return batch[0]

def get_dataloader(data_dir, batch_size, num_workers=12):
    dataset = FileLevelDataset(data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=raw_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

def plot_point_all_features(batch_x):
    sample_data = batch_x[0].detach().cpu()
    time_steps = sample_data.shape[0]
    num_points = 128
    num_freqs = 35 
    try:
        reshaped_data = sample_data.view(time_steps, num_points, num_freqs)
    except RuntimeError as e:
        print(f"Error during reshape: {e}. Ensure the last dim is 128*35=4480.")
        return
    rand_point_idx = random.randint(0, num_points - 1)
    print(f"Selected random point index: {rand_point_idx}")
    point_data_np = reshaped_data[:, rand_point_idx, :].numpy()
    fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(16, 14), sharex=True)
    axes_flat = axes.flatten() 
    
    for i in range(num_freqs):
        ax = axes_flat[i]
        ax.plot(point_data_np[:, i], linewidth=1, color='#1f77b4')
        ax.set_title(f"Feat {i}", fontsize=9, pad=3)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i % 5 == 0:
            ax.set_ylabel("Value", fontsize=8)

    fig.suptitle(f"Trajectory Visualization: Point Index {rand_point_idx} (All 35 Features shown independently)", 
                 fontsize=14, y=1.01)
    fig.text(0.5, 0.01, 'Time Steps', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig("features.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 4. Synthesis & Visualization Utils
# ==============================================================================
def synthesize_eeg_from_envelopes(gen_envelopes, start_phases, fs_high=500.0, f_start=4, f_end=38):
    """Memory Optimized: Synthesize EEG signals from envelopes."""
    device = gen_envelopes.device
    
    if start_phases.device != device:
        start_phases = start_phases.to(device)

    B, T_target, FeatDim = gen_envelopes.shape
    
    freqs = torch.arange(f_start, f_end + 1, device=device).float()
    n_freqs = len(freqs)
    n_channels = FeatDim // n_freqs
    
    env_4d = gen_envelopes.view(B, T_target, n_channels, n_freqs)
    phase_init_3d = start_phases.view(B, n_channels, n_freqs)
    
    synthetic_voltage = torch.zeros(B, T_target, n_channels, device=device)
    
    t = torch.linspace(0, (T_target - 1) / fs_high, T_target, device=device)
    t = t.view(1, T_target, 1) 
    
    for i, freq_val in enumerate(freqs):
        A_f = env_4d.select(-1, i) 
        phi_0_f = phase_init_3d.select(-1, i).unsqueeze(1)
        
        phase_t = 2 * np.pi * freq_val * t + phi_0_f
        
        component = torch.cos(phase_t)
        component.mul_(A_f) 
        
        synthetic_voltage.add_(component)
        
        del phase_t, component

    return synthetic_voltage

import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

def plot_psd_comparison(real_batch, y_labels, models_dict, sfreq, n_orig_ch, 
                        save_dir='./', filename_prefix="psd_comparison", 
                        f_min=4, f_max=38):
    """
    (Unified Style / 统一风格版)
    PSD 分析图：每个 Class 存一张独立的 PDF。
    风格：白底、灰网格、全黑边框 (Boxed)。
    """
    print(f"\n[Viz] Generating Clean PSD Plots -> {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1. 风格统一化 ---
    sns.set_theme(style="whitegrid", rc={
        "font.family": "serif",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "grid.color": "#E0E0E0",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False
    })

    # 配色方案
    STYLE_CONFIG = {
        "SSCI": {"color": "#D9363E", "lw": 2.0, "linestyle": "-", "zorder": 100, "alpha": 1.0, "label": "SSCI (Ours)"},
        "CSI": {"color": "#2E5EAA", "lw": 2.0, "linestyle": "--", "zorder": 90, "alpha": 1.0, "label": "CSI"},
        "LatentODEVaE": {"color": "#56A36C", "lw": 2.0, "linestyle": "--", "zorder": 10, "alpha": 1.0, "label": "LatentODEVaE"},
        "ConditionalTransformer": {"color": "#E09F3E", "lw": 2.0, "linestyle": "--", "zorder": 10, "alpha": 1.0, "label": "Cond. Trans."},
        "GaussianProcess": {"color":  "#7E57C2", "lw": 2.0, "linestyle": "--", "zorder": 5, "alpha": 1.0, "label": "GaussianProcess"}
    }
    
    # 数据预处理
    # (假设 restore_current_state 和 to_mne_format 在外部定义或导入)
    real_viz = restore_current_state(real_batch, n_orig_ch)
    real_viz = to_mne_format(real_viz)
    
    if y_labels.ndim > 1 and y_labels.shape[1] > 1: 
        y_indices = np.argmax(y_labels, axis=1)
    else: 
        y_indices = y_labels.astype(int)
        
    classes = np.unique(y_indices)
    
    for i, cls in enumerate(classes):
        mask = (y_indices == cls)
        if np.sum(mask) == 0: continue

        # --- 2. 尺寸调整 ---
        # 如果打算在 LaTeX 里放 4 张横排，单张图最好瘦高一点或者正方形
        # 如果是 2x2，可以稍微宽一点。这里设为 (4, 3.5) 比较灵活
        fig, ax = plt.subplots(figsize=(4, 3.5))
        
        # 计算 Real PSD
        psds, freqs = mne.time_frequency.psd_array_welch(
            real_viz[mask], sfreq=sfreq, fmin=f_min, fmax=f_max, 
            n_fft=256, verbose=False
        )
        psd_mean = 10 * np.log10(psds.mean(axis=(0, 1)) + 1e-12)
        
        # 绘图: Real Data
        ax.plot(freqs, psd_mean, 'k-', lw=2.5, label='Real Data', alpha=0.8, zorder=200)
        
        # 绘图: Models
        for name, tensor_data in models_dict.items():
            gen_clean = restore_current_state(tensor_data, n_orig_ch)
            gen_viz = to_mne_format(gen_clean)
            
            if len(gen_viz) == len(y_indices): 
                gen_viz = gen_viz[mask]
            
            if len(gen_viz) > 0:
                psds_gen, _ = mne.time_frequency.psd_array_welch(
                    gen_viz, sfreq=sfreq, fmin=f_min, fmax=f_max,
                    n_fft=256, verbose=False
                )
                psd_gen_mean = 10 * np.log10(psds_gen.mean(axis=(0, 1)) + 1e-12)
                
                if name in STYLE_CONFIG:
                    ax.plot(freqs, psd_gen_mean, **STYLE_CONFIG[name])
                else:
                    ax.plot(freqs, psd_gen_mean, label=name)

        # --- 3. 细节美化 ---
        ax.set_xlabel("Frequency (Hz)", fontsize=11)
        
        # 只有第一张图加 Y 轴标签，其他的为了节省空间可以省略（可选）
        if i == 0:
            ax.set_ylabel("Power Spectral Density (dB)", fontsize=11)
        else:
            ax.set_ylabel("") # 省略 Y 轴标题
            
        # 强制显示所有边框 (Boxed)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_visible(True)
            
        ax.grid(True)
        # 可以在图内右上角写个小小的 Class 标记，方便区分
        ax.text(0.95, 0.95, f"Class {cls}", transform=ax.transAxes, 
                ha='right', va='top', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # --- 4. 图例策略 ---
        # 只在第一张图 (Class 0) 或者最后一张图放图例
        # 这样 4 张横排的时候不会每张图都被图例挡住
        if i == 0: # 仅在第一个类别生成图例
            # 字体设小一点，framealpha=1.0 遮挡网格
            ax.legend(fontsize=8, loc='best', framealpha=1.0) 

        # 保存
        save_name = os.path.join(save_dir, f"{filename_prefix}_class{cls}.pdf")
        plt.tight_layout()
        plt.savefig(save_name, format='pdf', bbox_inches='tight')
        plt.close(fig)
        
        print(f"  -> Saved {save_name}")


def plot_heatmap_comparison(real_sample, models_sample_dict, n_orig_ch, 
                            save_dir='./', filename_prefix="heatmap_comparison", 
                            title_suffix="", duration=2.0): # <--- 新增 duration 参数
    """
    (TPAMI Style / 时间轴修正版)
    将 X 轴从 采样点(samples) 转换为 时间(seconds)。
    """
    print(f"\n[Viz] Generating Clean Heatmaps (Time-scaled) -> {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据准备
    real_clean = restore_current_state(real_sample, n_orig_ch)
    real_viz = to_mne_format(real_clean)
    
    models_viz = {}
    for k, v in models_sample_dict.items():
        v_clean = restore_current_state(v, n_orig_ch)
        models_viz[k] = to_mne_format(v_clean)
        
    vmin, vmax = get_global_scale(real_viz, models_viz)
    n_plots = 1 + len(models_viz)
    
    # 画布设置
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots), sharex=True, sharey=True)
    if n_plots == 1: axes = [axes]
    
    # 1. 画真实数据
    # imshow 默认是用像素索引 (0, 1, 2...) 作为坐标
    im = axes[0].imshow(real_viz, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title("") 
    axes[0].set_ylabel("Channels", fontsize=11)
    
    # 2. 画生成数据
    for i, (name, data) in enumerate(models_viz.items()):
        ax = axes[i+1]
        ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title("") 
        ax.set_ylabel("Channels", fontsize=11)
    
    # --- 核心修改：X轴 样本 -> 时间 映射 ---
    axes[-1].set_xlabel("Time (s)", fontsize=11) # 改名
    
    # 获取总采样点数 (例如 1000)
    n_samples = real_viz.shape[1] 
    
    # 生成 5 个等间距的时间点: [0.0, 0.5, 1.0, 1.5, 2.0]
    tick_times = np.linspace(0, duration, 5)
    
    # 计算这些时间点对应的 样本索引 (像素位置): [0, 250, 500, 750, 1000]
    tick_locs = np.linspace(0, n_samples - 1, 5)
    
    # 应用刻度
    axes[-1].set_xticks(tick_locs)
    axes[-1].set_xticklabels([f"{t:.1f}" for t in tick_times], fontsize=10)
    # -------------------------------------
    
    # 调整布局
    plt.subplots_adjust(hspace=0.1) 
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.3]) 
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Amplitude ($\mu$V)', fontsize=10)
    
    # 保存
    save_path = os.path.join(save_dir, f"{filename_prefix}.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"  -> Saved {filename_prefix}.pdf")
    plt.close(fig)
    
    
def animate_topomap_comparison(real_sample, models_sample_dict, info, sfreq, n_orig_ch, 
                               step=2, fps=60, t_max=None, dpi=200, save_path='comparison_result.mp4'):
    """Generates and saves a topomap animation."""
    print(f"\n[Viz] Rendering Topomap Animation (Step={step}, FPS={fps}, DPI={dpi})...")
    
    real_clean = restore_current_state(real_sample, n_orig_ch)
    real_viz = to_mne_format(real_clean)
    
    gen_viz_dict = {}
    for k, v in models_sample_dict.items():
        v_clean = restore_current_state(v, n_orig_ch)
        gen_viz_dict[k] = to_mne_format(v_clean)
    
    max_samples = real_viz.shape[1]
    if t_max: max_samples = min(max_samples, int(t_max * sfreq))
    
    frames = range(0, max_samples, step)
    vmin, vmax = get_global_scale(real_viz, gen_viz_dict)
    
    n_cols = 1 + len(gen_viz_dict)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))
    if n_cols == 1: axes = [axes]
    
    def update(frame_idx):
        axes[0].clear()
        mne.viz.plot_topomap(real_viz[:, frame_idx], info, axes=axes[0], show=False, vlim=(vmin, vmax), cmap='RdBu_r', contours=0, sensors=False)
        axes[0].set_title(f"Real Data\nT={frame_idx/sfreq:.2f}s")
        for i, (name, data) in enumerate(gen_viz_dict.items()):
            ax = axes[i+1]
            ax.clear()
            mne.viz.plot_topomap(data[:, frame_idx], info, axes=ax, show=False, vlim=(vmin, vmax), cmap='RdBu_r', contours=0, sensors=False)
            if name == 'SSCI':
                name = 'SSCI (Ours)'
            ax.set_title(f"{name}\n")
            
    anim = FuncAnimation(fig, update, frames=frames, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, bitrate=5000)
    
    try:
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"\n[Success] Animation saved to {save_path}")
    except Exception as e:
        print(f"\n[Error] Animation save failed: {e}")
        print("Retrying with lower DPI (100)...")
        anim.save(save_path, writer=writer, dpi=100)
    plt.close(fig)
    return anim


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    
    # --- Config ---
    setup_seed(seed=42)
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    
    if shutil.which("ffmpeg") is None:
        conda_ffmpeg = os.path.join(sys.prefix, 'Library', 'bin', 'ffmpeg.exe')
        if os.path.exists(conda_ffmpeg):
            plt.rcParams['animation.ffmpeg_path'] = conda_ffmpeg
            print(f"[Setup] Found ffmpeg in conda: {conda_ffmpeg}")
    
    
    TRAIN_SUBS = [1, 2, 3, 4, 5]
    TEST_SUBS = [6]
    FS_TRAIN = 100.0   # Low FS
    FS_VAL = 500.0     # High FS
    TRAIN_BUFFER_SEC = 1.0  
    HISTORY_SEC = 1.0  
    FUTURE_SEC = 2.0   
    LEN_TRAIN_TARGET = int(FUTURE_SEC * FS_TRAIN) + 1 
    LEN_VAL_TOTAL = int(HISTORY_SEC * FS_VAL) + int(FUTURE_SEC * FS_VAL) + 1
    ROOT_DIR = "prepared_eeg_symmetric"
    
    # ---------------------------------------------------------
    # A. Data Prep Pipeline
    # ---------------------------------------------------------
    if not os.path.exists(ROOT_DIR): 
        # 1. Run Train Phase
        run_train_phase(ROOT_DIR, TRAIN_SUBS, FS_TRAIN, LEN_TRAIN_TARGET, TRAIN_BUFFER_SEC, int(TRAIN_BUFFER_SEC * FS_TRAIN), HISTORY_SEC, FS_VAL)
        gc.collect()

        # 2. Run Val Phase
        run_val_phase(ROOT_DIR, TEST_SUBS, FS_VAL, FS_TRAIN, LEN_VAL_TOTAL, HISTORY_SEC)
        gc.collect()
        print("\n[Done] Pipeline finished (Symmetric Oracle Mode). Memory cleaned.")
    
    # ---------------------------------------------------------
    # B. Load Time Vector & Test DataLoader
    # ---------------------------------------------------------
    try:
        t_vector = get_train_time_vector(ROOT_DIR)
        sparse_t_list = t_vector # alias for model
    except Exception as e:
        print(f"Error loading time: {e}")
        exit()

    print(sparse_t_list)
    
    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    train_loader = get_dataloader(TRAIN_DIR, batch_size=1, num_workers=12)

    print("\n[Test] Iterating one batch...")
    for batch_x, batch_y in train_loader:
        print(f"Batch X (Envelopes): {batch_x.shape}") 
        print(f"Batch Y (Conditions): {batch_y.shape}") 
        print(f"Time Vector matches X? {batch_x.shape[1] == len(t_vector)}")
        plot_point_all_features(batch_x)
        break
        
    # Re-initialize loader for training
    train_loader = get_dataloader(TRAIN_DIR, batch_size=1, num_workers=12) # Typically batch_size depends on model logic, keeping 1 as per snippet

    # ---------------------------------------------------------
    # C. Model: Spline (SSCI)
    # ---------------------------------------------------------
    print("\n=== Training Spline Model (SSCI) ===")
    SI_class = model.Cubic_SI_model(
        train_loader, sparse_t_list, dataloader_input=True, d=128*35,
        n_layers=4, hiden_size=4096, concentrate=128*35,
        N_training=10, model_lr=1e-4, steps=1000,
        func_type='small', u_t=lambda x: 1e-2,
        spline=True, use_mlp=False,
        save_path='model_history/EEG_spline', record_gap=1, plot_loss=False
    )
    SI_class.train()

    # Inference SSCI
    VAL_DIR = os.path.join(ROOT_DIR, "val")
    chunk_files = sorted(glob.glob(os.path.join(VAL_DIR, "val_chunk_*.pt")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Inference SSCI] Found {len(chunk_files)} validation chunks.")
    buffer_synthesized, buffer_gt, buffer_conditions = [], [], []

    with torch.no_grad():
        for i, pth in enumerate(chunk_files):
            chunk_data = torch.load(pth)
            start_env = chunk_data['start_envelope'].to(device)
            start_phase = chunk_data['start_phase'].to(device)
            conditions = chunk_data['conditions'].to(device)
            
            generated_envelopes = SI_class.eval(start_env, conditions=conditions, SDE=True)
            generated_envelopes = torch.stack(generated_envelopes).transpose(0,1)
            
            syn_signal_gpu = synthesize_eeg_from_envelopes(generated_envelopes, start_phase, fs_high=500.0)
            
            buffer_synthesized.append(syn_signal_gpu.cpu())
            buffer_gt.append(chunk_data['gt_raw_voltage'])
            buffer_conditions.append(chunk_data['conditions'])
            
            del chunk_data, start_env, start_phase, conditions, generated_envelopes, syn_signal_gpu
            torch.cuda.empty_cache()
            if (i + 1) % 5 == 0: print(f"[Inference SSCI] Processed {i+1}/{len(chunk_files)}")

    final_signal_500hz_SSCI = torch.cat(buffer_synthesized, dim=0)
    # GT and Conditions are same for both models, we can keep them for final viz
    final_gt_voltage = torch.cat(buffer_gt, dim=0) 
    final_conditions = torch.cat(buffer_conditions, dim=0)
    
    del buffer_synthesized, buffer_gt, buffer_conditions
    gc.collect()

    # ---------------------------------------------------------
    # D. Final Visualization
    # ---------------------------------------------------------
    print("\n=== Final Visualization ===")
    
    # Prepare Dictionary
    models_full_batch = {
        # "CSI": final_signal_500hz_CSI,
        "SSCI": final_signal_500hz_SSCI,
    }
    
    try:
        META_FILE = os.path.join(ROOT_DIR, "dataset_meta.pt")
        DO_INVERSE_NORM = False
        TARGET_CLASS_ID = 1

        # Load Metadata
        meta = torch.load(META_FILE, map_location='cpu', weights_only=False)
        info = meta['info']
        sfreq = meta['sfreq_val']
        n_orig_ch = meta.get('original_channels', 128)

        # Prepare Real Data (from aggregated GT)
        real_full_batch = final_gt_voltage
        if isinstance(real_full_batch, torch.Tensor):
            real_full_batch = real_full_batch.numpy()
        val_labels = final_conditions
        if isinstance(val_labels, torch.Tensor): val_labels = val_labels.numpy()

        print(f"Metadata: sfreq={sfreq}Hz, Ch={n_orig_ch}")
        print(f"Real GT Shape: {real_full_batch.shape}")

        # Process Generated Tensors
        for name, data in models_full_batch.items():
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            if data.ndim != 3:
                print(f"[Error] {name} shape {data.shape} invalid. Skipping.")
                continue
            models_full_batch[name] = data
            print(f" -> {name} ready. Shape: {data.shape}")

        # Alignment
        print("\n[Pre-Viz] Checking Alignment & Norm...")
        target_T = real_full_batch.shape[1]
        for name, gen_data in models_full_batch.items():
            if gen_data.shape[1] != target_T:
                print(f"   Aligning {name}: {gen_data.shape[1]} -> {target_T}")
                models_full_batch[name] = resample_batch_to_target(gen_data, target_T)

        # Visualization
        plot_psd_comparison(real_full_batch, val_labels, models_full_batch, sfreq, n_orig_ch)

        print(f"\n[Viz] Selecting random sample from Class {TARGET_CLASS_ID}...")
        if val_labels.ndim > 1 and val_labels.shape[1] > 1:
            labels_flat = np.argmax(val_labels, axis=1)
        else:
            labels_flat = val_labels.astype(int)
        
        class_indices = np.where(labels_flat == TARGET_CLASS_ID)[0]
        if len(class_indices) > 0:
            sample_idx = np.random.choice(class_indices)
            print(f" -> Selected Sample Index: {sample_idx}")
        else:
            sample_idx = 0
            
        real_sample = real_full_batch[sample_idx]
        models_single = {k: v[sample_idx] for k, v in models_full_batch.items()}
        
        plot_heatmap_comparison(real_sample, models_single, n_orig_ch, title_suffix=f"(Class {TARGET_CLASS_ID}, Idx {sample_idx})")
        
        anim = animate_topomap_comparison(
            real_sample, models_single, info, sfreq, n_orig_ch,
            step=2, fps=60, t_max=2.0, dpi=200, save_path='comparison_high_res.mp4'
        )
        print("\n[Success] Animation saved to comparison_high_res.mp4")

    except Exception as e:
        import traceback
        traceback.print_exc()