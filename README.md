# EfficientUnet-PyTorch
A PyTorch 1.0 Implementation of Unet with EfficientNet as encoder

## Useful notes
1. Due to some rounding problem in the decoder path (*not a bug, this is a feature* :smirk:), the input shape should be divisible by 32.  
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
If you prefer to use a model with a custom head rather than just a simple change of the 
`output_channels` of the last fully-connected layer, use:
```python
from efficientunet import *

model = EfficientNet.custom_head('efficientnet-b5', n_classes=5, pretrained=True)
```
> *The structure of model with custom head*:  
`encoder` -> `concatenation of [AvgPool2d, MaxPool2d]` -> `Flatten` -> `Dropout` -> `Linear(512)` -> `ReLU` -> `Dropout`
> -> `Linear(n_classes)`

#### 2. EfficientUnets
e.g. say you want a *pretrained efficientunet-b0* model with *2* output channels:
```python
from efficientunet import *

b0unet = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True)
```

---
## Acknowledgment
The pretrained weights are directly borrowed from [this](https://github.com/lukemelas/EfficientNet-PyTorch) repo.
