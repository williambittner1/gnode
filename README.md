# Environment Setup for Project

This guide provides steps to set up the environment required to run this project, including configuring CUDA-11.7 and installing PyTorch.

---

## Prerequisites

1. **Operating System**: Tested on Linux (e.g., Ubuntu 20.04 or CentOS).
2. **CUDA Toolkit**: Ensure CUDA-11.7 is installed on your system.
3. **NVIDIA Driver**: Install the compatible NVIDIA driver for CUDA-11.7. You can find the compatibility list [here](https://docs.nvidia.com/deploy/cuda-compatibility/).
4. **Python**: Python 3.8+ is recommended.
5. **Conda**: Miniconda or Anaconda installed.

---

## Setting Up the Environment

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

### Step 2: Configure CUDA-11.7

Check if CUDA-11.7 is installed:

```bash
ls /usr/local | grep cuda-11.7
Ensure the bin and lib64 directories exist in /usr/local/cuda-11.7.
```

Add CUDA-11.7 to the PATH and LD_LIBRARY_PATH:

```bash
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

Verify the setup:

```bash
nvcc --version
```

### Step 3: Set Up Python Environment

Create a Conda Environment:

```bash
conda create --name gnode python=3.8 -y
conda activate gnode
```

### Step 4: Install Pytorch with CUDA-11.7

Install PyTorch: Install the specific version of PyTorch compatible with CUDA-11.7:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

The output should show Cuda compilation tools, release 11.7.

### Step 5: Install Gaussian Splatting Rasterizer

```bash
git clone <gaussian_splatting_directory>
cd <gaussian_splatting_directory>
pip install submodules/diff-gaussian-rasterization
```
pip install submodules/simple-knn
