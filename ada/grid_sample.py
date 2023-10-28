import torch.nn.functional as F

def grid_sample(input, grid):
    return F.grid_sample(input=input, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=False)
