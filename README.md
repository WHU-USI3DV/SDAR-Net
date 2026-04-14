# SDAR-Net
This repository contains the official implementation of the paper Style-Decoupled Adaptive Routing Network for Underwater Image Enhancement

## Get Started

### Requirement
- python == 3.9
- PyTorch == 2.4.0
- CUDA == 11.8
- ninja == 1.11.1
- einops
- diffusers == 0.36.0
- tqdm
- numpy
- matplotlib

The code has been tested on Ubuntu 20.04 with NVIDIA GeForce RTX 3080 Ti, 4090 and 5090 D. Its dependences version is not very strict, but I recommend at least keeping CUDA==11.8.

### Dataset
We provide the divided dataset for UIEB and LSUI. This division refers to [WF-diff](https://github.com/ChenzhaoNju/WF-Diff)
   
| UIEB | [UIEB](  https://pan.baidu.com/s/1BWtIPz-xUDaatsncOFCJHg?pwd=123x  ) | 

| LSUI | [LSUI](   https://pan.baidu.com/s/1-Nk8iqmOVIl3ulZTHkdpbQ?pwd=123x  ) | 

Extract code:123x

### Evaluation
```
# Specify the dataset path and checkpoint path in test.py

python test.py
```

### Training
```
# Specify the dataset path and checkpoint path in train.py

python train.py
```

## Acknowledgement
The repository is based on [U-Shape](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement), some of the code is borrowed from:
- [Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR)

Thanks for their opensourceing.
