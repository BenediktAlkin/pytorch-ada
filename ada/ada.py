import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch import nn

from .constant import constant
from .matrix_utils import (
    matrix,
    translate2d,
    translate3d,
    scale2d,
    scale2d_inv,
    scale3d,
    rotate2d_inv,
    rotate3d,
    translate2d_inv,
)
from .upfirdn2d import downsample2d, upsample2d, setup_filter
from .wavelets import wavelets


class AdaptiveDiscriminatorAugmentation(nn.Module):
    def __init__(
            self,
            xflip=0,
            rotate90=0,
            xint=0,
            xint_max=0.125,
            scale=0,
            rotate=0,
            aniso=0,
            xfrac=0,
            scale_std=0.2,
            rotate_max=1,
            aniso_std=0.2,
            xfrac_std=0.125,
            brightness=0,
            contrast=0,
            lumaflip=0,
            hue=0,
            saturation=0,
            brightness_std=0.2,
            contrast_std=0.5,
            hue_max=1,
            saturation_std=1,
            imgfilter=0,
            imgfilter_bands=(1, 1, 1, 1),
            imgfilter_std=1,
            noise=0,
            cutout=0,
            noise_std=0.1,
            cutout_size=0.5,
    ):
        super().__init__()
        # Overall multiplier for augmentation probability.
        self.register_buffer("p", torch.tensor(1.))

        # Pixel blitting.
        # Probability multiplier for x-flip.
        self.xflip = float(xflip)
        # Probability multiplier for 90 degree rotations.
        self.rotate90 = float(rotate90)
        # Probability multiplier for integer translation.
        self.xint = float(xint)
        # Range of integer translation, relative to image dimensions.
        self.xint_max = float(xint_max)

        # General geometric transformations.
        # Probability multiplier for isotropic scaling.
        self.scale = float(scale)
        # Probability multiplier for arbitrary rotation.
        self.rotate = float(rotate)
        # Probability multiplier for anisotropic scaling.
        self.aniso = float(aniso)
        # Probability multiplier for fractional translation.
        self.xfrac = float(xfrac)
        # Log2 standard deviation of isotropic scaling.
        self.scale_std = float(scale_std)
        # Range of arbitrary rotation, 1 = full circle.
        self.rotate_max = float(rotate_max)
        # Log2 standard deviation of anisotropic scaling.
        self.aniso_std = float(aniso_std)
        # Standard deviation of frational translation, relative to image dimensions.
        self.xfrac_std = float(xfrac_std)

        # Color transformations.
        # Probability multiplier for brightness.
        self.brightness = float(brightness)
        # Probability multiplier for contrast.
        self.contrast = float(contrast)
        # Probability multiplier for luma flip.
        self.lumaflip = float(lumaflip)
        # Probability multiplier for hue rotation.
        self.hue = float(hue)
        # Probability multiplier for saturation.
        self.saturation = float(saturation)
        # Standard deviation of brightness.
        self.brightness_std = float(brightness_std)
        # Log2 standard deviation of contrast.
        self.contrast_std = float(contrast_std)
        # Range of hue rotation, 1 = full circle.
        self.hue_max = float(hue_max)
        # Log2 standard deviation of saturation.
        self.saturation_std = float(saturation_std)

        # Image-space filtering.
        # Probability multiplier for image-space filtering.
        self.imgfilter = float(imgfilter)
        # Probability multipliers for individual frequency bands.
        self.imgfilter_bands = list(imgfilter_bands)
        # Log2 standard deviation of image-space filter amplification.
        self.imgfilter_std = float(imgfilter_std)

        # Image-space corruptions.
        # Probability multiplier for additive RGB noise.
        self.noise = float(noise)
        # Probability multiplier for cutout.
        self.cutout = float(cutout)
        # Standard deviation of additive RGB noise.
        self.noise_std = float(noise_std)
        # Size of the cutout rectangle, relative to image dimensions.
        self.cutout_size = float(cutout_size)

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer("Hz_geom", setup_filter(wavelets["sym6"]))

        # Construct filter bank for image-space filtering.
        # H(z)
        Hz_lo = np.asarray(wavelets["sym2"])
        # H(-z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))
        # H(z) * H(z^-1) / 2
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2
        # H(-z) * H(-z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2
        # Bandpass(H(z), b_i)
        Hz_fbank = np.eye(4, 1)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2: (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.register_buffer("Hz_fbank", torch.as_tensor(Hz_fbank, dtype=torch.float32))

    def set_p(self, p):
        if not torch.is_tensor(p):
            p = torch.tensor(p)
        self.p.copy_(p)

    def forward(self, images):
        assert torch.is_tensor(images) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 2)
            i = torch.where(torch.rand([batch_size], device=device) < self.xflip * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = torch.floor(torch.rand([batch_size], device=device) * 4)
            i = torch.where(torch.rand([batch_size], device=device) < self.rotate90 * self.p, i, torch.zeros_like(i))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (torch.rand([batch_size, 2], device=device) * 2 - 1) * self.xint_max
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xint * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(torch.round(t[:, 0] * width), torch.round(t[:, 1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.scale_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.scale * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = 1 - torch.sqrt((1 - self.rotate * self.p).clamp(0, 1))  # P(pre OR post) = p
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta)  # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            s = torch.exp2(torch.randn([batch_size], device=device) * self.aniso_std)
            s = torch.where(torch.rand([batch_size], device=device) < self.aniso * self.p, s, torch.ones_like(s))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.rotate_max
            theta = torch.where(torch.rand([batch_size], device=device) < p_rot, theta, torch.zeros_like(theta))
            G_inv = G_inv @ rotate2d_inv(-theta)  # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = torch.randn([batch_size, 2], device=device) * self.xfrac_std
            t = torch.where(torch.rand([batch_size, 1], device=device) < self.xfrac * self.p, t, torch.zeros_like(t))
            G_inv = G_inv @ translate2d_inv(t[:, 0] * width, t[:, 1] * height)

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        # Execute if the transform is not identity.
        if G_inv is not I_3:
            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device)  # [idx, xyz]
            cp = G_inv @ cp.t()  # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1)  # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values  # [x0, y0, x1, y1]
            margin = margin + constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(constant([0, 0] * 2, device=device))
            margin = margin.min(constant([width - 1, height - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = F.pad(input=images, pad=[mx0, mx1, my0, my1], mode="reflect")
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = (
                    scale2d(2 / images.shape[3], 2 / images.shape[2], device=device)
                    @ G_inv
                    @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            )
            grid = F.affine_grid(theta=G_inv[:, :2, :], size=shape, align_corners=False)
            images = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

            # Downsample and crop.
            images = downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad * 2, flip_filter=True)

        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_4 = torch.eye(4, device=device)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            b = torch.randn([batch_size], device=device) * self.brightness_std
            b = torch.where(torch.rand([batch_size], device=device) < self.brightness * self.p, b, torch.zeros_like(b))
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (contrast * strength).
        if self.contrast > 0:
            c = torch.exp2(torch.randn([batch_size], device=device) * self.contrast_std)
            c = torch.where(torch.rand([batch_size], device=device) < self.contrast * self.p, c, torch.ones_like(c))
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (lumaflip * strength).
        v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3), device=device)  # Luma axis.
        if self.lumaflip > 0:
            i = torch.floor(torch.rand([batch_size, 1, 1], device=device) * 2)
            i = torch.where(
                torch.rand([batch_size, 1, 1], device=device) < self.lumaflip * self.p,
                i,
                torch.zeros_like(i),
            )
            C = (I_4 - 2 * v.ger(v) * i) @ C  # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0 and num_channels > 1:
            theta = (torch.rand([batch_size], device=device) * 2 - 1) * np.pi * self.hue_max
            theta = torch.where(
                torch.rand([batch_size], device=device) < self.hue * self.p,
                theta,
                torch.zeros_like(theta),
            )
            C = rotate3d(v, theta) @ C  # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0 and num_channels > 1:
            s = torch.exp2(torch.randn([batch_size, 1, 1], device=device) * self.saturation_std)
            s = torch.where(
                torch.rand([batch_size, 1, 1], device=device) < self.saturation * self.p,
                s,
                torch.ones_like(s),
            )
            C = (v.ger(v) + (I_4 - v.ger(v)) * s) @ C

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            if num_channels == 3:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
            elif num_channels == 1:
                C = C[:, :3, :].mean(dim=1, keepdims=True)
                images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
            else:
                raise ValueError("Image must be RGB (3 channels) or L (1 channel)")
            images = images.reshape([batch_size, num_channels, height, width])

        # ----------------------
        # Image-space filtering.
        # ----------------------

        if self.imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands
            # Expected power spectrum (1/f).
            expected_power = constant(np.array([10, 1, 1, 1]) / 13, device=device)

            # Apply amplification for each band with probability (imgfilter * strength * band_strength).
            g = torch.ones([batch_size, num_bands], device=device)  # Global gain vector (identity).
            for i, band_strength in enumerate(self.imgfilter_bands):
                t_i = torch.exp2(torch.randn([batch_size], device=device) * self.imgfilter_std)
                t_i = torch.where(
                    torch.rand([batch_size], device=device) < self.imgfilter * self.p * band_strength,
                    t_i,
                    torch.ones_like(t_i),
                )
                t = torch.ones([batch_size, num_bands], device=device)  # Temporary gain vector.
                t[:, i] = t_i  # Replace i"th element.
                t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt()  # Normalize power.
                g = g * t  # Accumulate into global gain.

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank  # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])  # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1])  # [batch * channels, 1, tap]

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            images = images.reshape([1, batch_size * num_channels, height, width])
            images = F.pad(input=images, pad=[p, p, p, p], mode="reflect")
            images = F.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size * num_channels)
            images = F.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size * num_channels)
            images = images.reshape([batch_size, num_channels, height, width])

        # ------------------------
        # Image-space corruptions.
        # ------------------------

        # Apply additive RGB noise with probability (noise * strength).
        if self.noise > 0:
            sigma = torch.randn([batch_size, 1, 1, 1], device=device).abs() * self.noise_std
            sigma = torch.where(
                torch.rand([batch_size, 1, 1, 1], device=device) < self.noise * self.p,
                sigma,
                torch.zeros_like(sigma),
            )
            images = images + torch.randn([batch_size, num_channels, height, width], device=device) * sigma

        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            size = torch.full([batch_size, 2, 1, 1, 1], self.cutout_size, device=device)
            size = torch.where(
                torch.rand([batch_size, 1, 1, 1, 1], device=device) < self.cutout * self.p,
                size,
                torch.zeros_like(size),
            )
            center = torch.rand([batch_size, 2, 1, 1, 1], device=device)
            coord_x = torch.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = torch.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = torch.logical_or(mask_x, mask_y).to(torch.float32)
            images = images * mask

        return images
