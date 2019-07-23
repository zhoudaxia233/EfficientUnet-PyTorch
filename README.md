# EfficientUnet-PyTorch
A PyTorch 1.0 Implementation of Unet with EfficientNet as encoder

## Useful notes
1. Due to some rounding problem in PyTorch (*not so sure*), the input shape should be divisible by 32.  
e.g. 224x224 is a suitable size for input images, but 225x225 is not.

---
## Requirements
1. Python >= 3.6
2. [PyTorch](https://pytorch.org/get-started/locally/) >= 1.0.0

---
## Installation
Install `efficientunet-pytorch`:
```bash
pip install efficientunet-pytorch
```

---
## Usage
#### 1. EfficientNets
e.g. say you want a *pretrained efficientnet-b5* model with *5* classes:
```python
from efficientunet import *

model = EfficientNet.from_name('efficientnet-b5', n_classes=5, pretrained=True)
```
#### 2. EfficientUnets
e.g. say you want a *pretrained efficientunet-b0* model with *2* output channels:
```python
from efficientunet import *

b0unet = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True)
```

---
## Acknowledgment
The pretrained weights are directly borrowed from [this](https://github.com/lukemelas/EfficientNet-PyTorch) repo.
