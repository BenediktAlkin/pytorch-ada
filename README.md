# ImageNetUtils

Utilities for analyzing ImageNet classes.

# Setup
`pip install git+https://github.com/BenediktAlkin/pytorch-ada`

# Usage

```python
import torch
from ada import AdaptiveDiscriminatorAugmentation

# create augmentation pipeline
aug = AdaptiveDiscriminatorAugmentation(
    xflip=1
)
# create 4 RGB images with 32x32 resolution
x = torch.randn(4, 3, 32, 32)
# augment images
augmented = aug(x)
# adjust strength
aug.p = 0.5
```
