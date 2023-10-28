import torch

from reference.augment import AugmentPipe
from ada import AdaptiveDiscriminatorAugmentation

def run(aug):
    torch.manual_seed(0)
    x = torch.rand(size=(32, 3, 224, 224))
    return aug(x)

def main():
    kwargs = dict(
            xflip=1,
            rotate90=1,
            xint=1,
            xint_max=0.125,
            scale=1,
            rotate=1,
            aniso=1,
            xfrac=1,
            scale_std=0.2,
            rotate_max=1,
            aniso_std=0.2,
            xfrac_std=0.125,
            brightness=1,
            contrast=1,
            lumaflip=1,
            hue=1,
            saturation=1,
            brightness_std=0.2,
            contrast_std=0.5,
            hue_max=1,
            saturation_std=1,
            imgfilter=1,
            imgfilter_bands=(1, 1, 1, 1),
            imgfilter_std=1,
            noise=1,
            cutout=1,
            noise_std=0.1,
            cutout_size=0.5,
    )
    aug0 = AugmentPipe(**kwargs)
    aug1 = AdaptiveDiscriminatorAugmentation(**kwargs)

    y0 = run(aug0)
    y1 = run(aug1)

    assert torch.all(y0 == y1)
    print("fin")


if __name__ == '__main__':
    main()