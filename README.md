# [Project Name]

Official implementation for the paper **"Spline based Stochastic Conditional Interpolation for Dynamics Reconstruction and Trajectory Generation"**.

In this work, we aim to comprehensively evaluate the effectiveness and robustness of our approach in reconstructing underlying dynamics mechanisms from sparse snapshots and generating new trajectories given initial values and conditions or controlling signals. This repository contains the source code, simulated environments, and scripts to reproduce the sequence of experiments covering different dynamics settings presented in our paper.

## 📂 Repository Structure

* **`cubic_SI/`**: Core library containing the model architectures, numerical solvers, and utility functions for system identification.
* **`torchcubicspline/`**: Included dependency for cubic spline interpolations in PyTorch.
* **`Particle_Electromagnetic_Motion.ipynb`**: Notebook for the heterogeneous particle electromagnetic motion experiment.
* **`Stochastic_Swarm_Dynamics.ipynb`**: Notebook for the stochastic swarm dynamics experiment.
* **`Kinematic_Reconstruction.ipynb`**: Notebook for the full-body pose reconstruction experiment (animations and filmstrip plots included).
* **`EEG_Signals.py`**: Main script for modeling and generating high-dimensional EEG signals.

## 🛠️ Environment Setup

This project depends on `PyTorch`, standard scientific computing libraries, Optimal Transport packages, and domain-specific tools for EEG and human body modeling (SMPLX/Open3D). We recommend using Anaconda to manage the environment.

```bash
# Clone the repository
git clone [https://github.com/your_username/](https://github.com/your_username/)[Project_Name].git
cd [Project_Name]

# Create a virtual environment (Python 3.8+ recommended)
conda create -n dynamics_env python=3.9
conda activate dynamics_env

# Install PyTorch (Verify your CUDA version at [https://pytorch.org/](https://pytorch.org/))
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Data Science and ML standard libraries
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm jupyter

# Install Optimal Transport (OT) packages for Wasserstein distance calculation
pip install POT geomloss

# Install EEG processing libraries
pip install mne braindecode

# Install 3D rendering and animation tools for Kinematic experiments
pip install smplx open3d imageio
```

> **Note**: The `torchcubicspline` package is included directly in this repository to ensure compatibility for trajectory interpolation. You do not need to install it separately.

## 💾 Data Preparation

The data requirements vary depending on the specific experiment:

1.  **Physical Simulation Benchmarks (`Particle_Electromagnetic_Motion.ipynb`, `Stochastic_Swarm_Dynamics.ipynb`)**
    * No data download is required. The initial values and environments are purely simulated and generated on-the-fly during runtime.
2.  **High-dimensional EEG Signals (`EEG_Signals.py`)**
    * No manual download is required. The script utilizes `braindecode` to automatically fetch and preprocess the public high-gamma dataset (HGD) upon first run.
3.  **Kinematic Reconstruction (`Kinematic_Reconstruction.ipynb`)**
    * This experiment utilizes human motion capture data. We are currently packaging the standardized dataset.
    * *Coming Soon:* A download link (e.g., Google Drive/Baidu Netdisk) will be provided here shortly. Once available, please download and extract it into a `Data/` folder in the root directory.

## 🚀 Usage & Reproduction

We designed a sequence of experiments covering different dynamics settings. You can reproduce the models and evaluations using the provided scripts and notebooks:

### 1. Physical Simulation Benchmarks
These experiments evaluate the model on both conventional ODEs and high-dimensional SDEs.
* **`Particle_Electromagnetic_Motion.ipynb`**: Evaluates the model on **heterogeneous particle electromagnetic motion** (Conventional ODE benchmark).
* **`Stochastic_Swarm_Dynamics.ipynb`**: Evaluates the model on **stochastic swarm dynamics** (High-dimensional SDE benchmark).

### 2. Real-World Case Studies
These experiments extend complex dynamic modeling to real-world applications and control signals.
* **`Kinematic_Reconstruction.ipynb`**: Performs **full-body pose reconstruction** from simulated VR signals. This serves as an ODE generating task under dynamic control sequences. Outputs include 3D animations and sequence filmstrip visualizations.
* **`EEG_Signals.py`**: Runs the task for generating high-dimensional **EEG signals**, modeled as complex SDE processes.
    * *Usage*: `python EEG_Signals.py`

## 🔗 Citation

If you find this code or data useful for your research, please cite our paper:

```bibtex
@article{YourPaper202X,
  title={[Your Paper Title]},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={202X},
  publisher={Publisher}
}
```

## 🙏 Acknowledgements

* This codebase incorporates components from `torchcubicspline` for continuous trajectory modeling.
* We utilize `braindecode` and `mne` for EEG data processing, and `smplx` alongside `open3d` for 3D kinematic rendering. We thank the authors of these open-source tools for their contributions.
