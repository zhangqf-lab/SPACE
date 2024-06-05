[![Stars](https://img.shields.io/github/stars/zhangqf-lab/SPACE?logo=GitHub&color=yellow)](https://github.com/zhangqf-lab/SPACE) [![PyPI](https://img.shields.io/pypi/v/space-srt.svg)](https://pypi.org/project/space-srt)

# Tissue module discovery in single-cell-resolution spatial transcriptomics data via cell-cell interaction-aware cell embedding
SPACE (**SP**atial transcriptomics **A**nalysis via **C**ell **E**mbedding)

## Installation  	

SPACE is implemented in [Pytorch](https://pytorch.org/) framework.  
SPACE can be run on CPU devices, and running SPACE on GPU devices if available is recommended.   

### Software dependencies
#### Install Pytorch
Please install Pytorch in advance by following the instructions on : https://pytorch.org/get-started/locally/

#### Install PyTorch Geometric
Please install PyTorch Geometric in advance by following the instructions on : https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html


### Install SPACE from PyPI

```shell
pip install space-srt
```

### Install SPACE from GitHub

install the latest develop version

```shell
git clone https://github.com/zhangqf-lab/SPACE.git
cd SPACE
python setup.py install
```

## Tutorial
A brief tutorial can be found [here](https://tutorial-space.readthedocs.io/en/latest/).（Still in progress）

## Citation
If you use SPACE in your research, please cite our paper:

Li, Y., Zhang J., Gao, X., and Zhang, Q.C. Tissue module discovery in single-cell resolution spatial transcriptomics data via cell- cell interaction-aware cell embedding. Cell Systems 2024 ([paper](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00124-8?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2405471224001248%3Fshowall%3Dtrue))
