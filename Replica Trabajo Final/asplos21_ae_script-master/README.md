# System Pre-requisite

* Ubuntu 18.04.5 LTS Linux
* git
* Python 2.7
* pyyaml
* CUDA 10.1: https://developer.nvidia.com/cuda-10.1-download-archive-base

# Install pyyaml
Our artifact repo request to use pyyaml to parse configurations, you can download with pip:
```bash
pip install pyyaml
```

# Install CUDA 10.1

Download and install CUDA 10.1(If you have CUDA 10.1 installed, you can skip this part):
```bash
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sh cuda_10.1.243_418.87.00_linux.run
```

# Enable to access NVIDIA GPU performance counters to use NVProf

NVIDIA disable to use performance counters for security reasons: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters. To use NVProf from CUDA 10.1 to profile the application, You need to enable the counters.


# Configure system environment

Configure system environment for CUDA 10.1 installed path:
```bash
export CUDA_INSTALL_PATH=<cuda-toolkit-path>
```

Setup envs based on CUDA intalled path:
```bash
source setup.sh
```

# Clone repo and compile workloads

Clone github repo and compile the workloads:
```bash
source compile.sh
```

# Run workloads

Run all the workloads with device 0 (and you can change 0 to a different device number), and it takes about 40 mins on V100 GPUs:
```bash
source run.sh 0
```

# Get statistics from workloads

Get all the workloads data with device 0 (and you can change 0 to a different device number):
```bash
source get.sh 0
```

The output will be like below:
```
  trafficV 0.443328212062
  trafficV_CONCORD 0.469073964166
  trafficV_MEM 1.0
  ...
  RAY_COAL 0.945034105267
  RAY_TP 0.935876022056
```

The output name for workloads and techniques are little different, so we attach two tables below:

Set | Workload name | Output name
----|----|----
Dynasoar|TRAF|trafficV
____|GOL|game-of-life
____|STUT|structureV
____|GEN|generationV
GraphChi-VE|BFS|BFS
____|CC|CC
____|PR|PR
GraphChi-VEN|BFS|BFSV
____|CC|CCV
____|PR|PRV
Raytracer|RAY|RAY

Techniques|Output suffix name
----|----
CUDA|no suffix
Concord|_CONCORD
SharedOA|_MEM
SharedOA+COAL|_COAL
SharedOA+TypePointer|_TP

For example, structure workloads with SharedOA techniques is named "STUT_MEM" and game-of-life with CUDA is named "GOL".

# Remove old log for the next experiment
You need to remove the log from the previous test so that you can start for the next experiment:
```bash
rm -rf asplos_2021_ae/run_hw/device-*
```
