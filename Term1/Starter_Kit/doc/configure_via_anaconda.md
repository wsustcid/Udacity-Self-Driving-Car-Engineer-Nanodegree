# Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer
2. Create a new `conda` [environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using this project
3. Each time you wish to work, activate your `conda` environment

---

## Installation

**1. Download** the latest version of `miniconda` that matches your system.

**NOTE**: 

- There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).
- By my test, many other versions, such as `v4.8.2`, `4.7`,`4.3.11` also failed to create environments, so it’s better to use try another version (`v4.6.14`) when you encounter various wried errors.
- The network condition often causes various installation errors when conda creates environments, I have no solutions to solve this problem except to try creating many times.
- 所以在安装的时候，报错的时候你要判断是由于版本问题导致的错误还是由于网络问题导致的无法安装。。。。
- 经过无数次的失败尝试，觉得主要的问题是Udacity的环境配置文件的依赖指定了版本(比如python和tensorflow)，有的没有，没有制定的时候会默认装最新版，这样就会和指定的一些旧版本的依赖发生冲突。所以我最后基于conda的v4.6.14版本，修改了环境配置文件，去掉了python版本指定，tensorflow和keras选用了我常用的版本(和cuda9同一时期的，因为显卡驱动是381)

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh



**2. Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

  ```python
  # 1. In your terminal window, run:
  #Miniconda:
  bash Miniconda3-latest-Linux-x86_64.sh
  #Anaconda:
  bash Anaconda-latest-Linux-x86_64.sh
  # 2. Follow the prompts on the installer screens. ENTER->yes->ENTER->
  # 3. To make the changes take effect, close and then re-open your terminal window.
  # 4. If you'd prefer that conda's base environment not be activated on startup, set the auto_activate_base parameter to false: 
  # for conda 4.82
  conda config --set auto_activate_base false
  # 
  # 5. Test your installation. In your terminal window or Anaconda Prompt, run the command 
  conda list
  # packages in environment at /home/ubuntu16/miniconda3:
  #
  # Name                    Version                   Build  Channel
  _libgcc_mutex             0.1                        main  
  ...
  ```

- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install

- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install



**3. Setup** the `carnd-term1` environment. 

```sh
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
cd CarND-Term1-Starter-Kit
# For my project, is cd Udacity-Self-Driving-Car-Engineer-Nanodegree/Term1/Starter_Kit/
```

If you are on Windows, **rename**   `meta_windows_patch.yml` to `meta.yml`



**4. Create** carnd-term1.  Running this command will create a new `conda` environment that is provisioned with all libraries you need to be successful in this program.

```python
conda env create -f environment.yml
#or
conda env create -f environment-gpu.yml
# for my local computer:
conda env create -f environment-cuda9-tf1.9.0.yml
```

***Note*:** 

- Some Mac users have reported issues installing TensorFlow using this method. The cause is unknown but seems to be related to `pip`. For the time being, we recommend opening environment.yml in a text editor and swapping

```yaml
    - tensorflow==0.12.1
```
with
```yaml
    - https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
```
If you have encountered a No module named 'requests' error, try to add in a line under 'pip' line in the environment.yml in a text editor

with
```yaml
    - requests
```

**5. Verify** that the carnd-term1 environment was created in your environments:

```sh
conda info --envs
```

**6. Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

### Uninstalling 

To uninstall the environment:

```sh
conda env remove -n carnd-term1
```

---

## GPU Installation

Prior to installing tensorflow-gpu for Ubuntu or Windows as part of the Anaconda environment for Nvidia GPUs, install the appropriate versions of CUDA Toolkit and cuDNN, along with the necessary Nvidia drivers. See Ubuntu instructions [here](https://www.tensorflow.org/install/install_linux) and Windows instructions [here](https://www.tensorflow.org/install/install_windows).

When creating the environment, at the **Create** step above, change the command to:
```
conda env create -f environment-gpu.yml
```
Otherwise, follow the same steps as above.

---

## Using Anaconda

Now that you have created an environment, in order to use it, you will need to activate the environment. This must be done **each** time you begin a new working session i.e. open a new terminal window. 

**Activate** the `carnd-term1` environment:

### OS X and Linux
```sh
$ source activate carnd-term1

$ source deactivate

# for newer conda version
conda activate xx
```
### Windows
Depending on shell either:
```sh
$ source activate carnd-term1
```
or

```sh
$ activate carnd-term1
```

That's it. Now all of the `carnd-term1` libraries are available to you.

To exit the environment when you have completed your work session, simply close the terminal window.

