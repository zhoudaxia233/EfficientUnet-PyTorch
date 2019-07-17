# EfficientUnet-PyTorch
A PyTorch 1.0 Implementation of Unet with EfficientNet as encoder

## Useful notes
1. Due to some rounding problem in PyTorch (*not so sure*), the input shape should be divisible by 32.  
e.g. 224x224 is a suitable size for input images, but 225x225 is not.
