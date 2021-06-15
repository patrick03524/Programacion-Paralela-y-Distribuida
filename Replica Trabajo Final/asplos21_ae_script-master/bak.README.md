# Install conda

Instructions below follows [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#). We just need to install conda once.

Download conda package:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
```

Install conda:
```bash
bash Anaconda3-2020.11-Linux-x86_64.sh 
```

Enable bash shell with conda:
```bash
cd ~/anaconda3/bin
conda init bash
```

re-open the terminal and enable conda command:
```bash
. .bashrc
```

Create conda env with python 2.7
```bash
conda create --name py2.7 python=2.7
```

activate py2.7
```bash
conda activate py2.7
```
