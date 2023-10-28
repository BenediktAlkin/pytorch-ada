# Adaptive Discriminator Augmentation for pytorch

Standalone implementation of [Adaptive Discriminator Augmentation](https://arxiv.org/abs/2006.06676). 
Based on the [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repository.

# Setup
`pip install git+https://github.com/BenediktAlkin/pytorch-ada`

# Usage

```python
import torch
from ada import AdaptiveDiscriminatorAugmentation

# create augmentation pipeline
aug = AdaptiveDiscriminatorAugmentation(
    xflip=1, 
    rotate90=1,
    xint=1, 
    scale=1, 
    rotate=1, 
    aniso=1,
    xfrac=1, 
    brightness=1, 
    contrast=1, 
    lumaflip=1,
    hue=1, 
    saturation=1,
)
# create 4 RGB images with 32x32 resolution
x = torch.randn(4, 3, 32, 32)
# augment images
augmented = aug(x)
# adjust strength
aug.set_p(0.5)
```
