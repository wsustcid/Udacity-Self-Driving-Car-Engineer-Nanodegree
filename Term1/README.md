# **Prerequisites for Term 1** 
## 方案1：Starter Kit Installation

**Step 1:**  Follow the instructions and install the [CarND Term1 Starter Kit](./term1_starter_kit/README.md) 

```python
## install miniconda (安装后无法安装包，待解决)
# 1. Download the latest version of miniconda
# 2. install: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent
bash Miniconda3-latest-Linux-x86_64.sh
# 3. close and then re-open your terminal window.
(base) ubuntu16@Laptop1:~$
conda list.
# packages in environment at /home/ubuntu16/miniconda3:
# 然后注释掉~bashrc中conda相关的命令，防止干扰其他操作，使用时激活即可
source miniconda3/bin/activate

## install NVIDIA Drivers and CUDA for tensorflow-gpu

## Setup the `carnd-term1` environment.
cd Self-Driving-Projects/Term1/term1_starter_kit/
conda env create -f environment-gpu.yml
```



**Step 2:** Open the code in a Jupyter Notebook for each project. Be sure you've activated your Python 3 carnd-term1 environment.

```python

```



## 方案2：使用virtualenv 手动安装相关包

```python
## 基本环境准备
# 1. 安装NVIDIA显卡驱动
# 2. 安装驱动对应版本的CUDA,cudnn
# 3. 创建虚拟环境
sudo apt-get install python3-pip python3-dev python-virtualenv
virtualenv --no-site-packages -p python3 venv
source ~/venv/bin/activate

## 安装term1依赖包
    - python==3.5.2
    - numpy
    - matplotlib
    - jupyter
    - pillow
    - scikit-learn
    - scikit-image
    - scipy
    - h5py
    - eventlet
    - flask-socketio
    - seaborn
    - pandas
    - ffmpeg
    - imageio
    - pyqt=4.11.4  # pip install pyqt5
    - pip:
        - moviepy
        - opencv-python
        - requests
        - tensorflow_gpu
        - keras==2.0.9
        
 # 解决pip安装速度慢的问题
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple
```

