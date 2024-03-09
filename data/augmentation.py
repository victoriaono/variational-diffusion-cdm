import torch
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import Resize as VisionResize

class Normalize(torch.nn.Module):
    def __init__(
        self,
        mean_input,
        std_input,
        mean_target,
        std_target,
    ):
        super().__init__()
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target = mean_target
        self.std_target = std_target

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        conditioning, target = inputs
        transformed_conditioning = F.normalize(
            conditioning,
            self.mean_input,
            self.std_input,
        )
        transformed_target = F.normalize(
            target,
            self.mean_target,
            self.std_target,
        )
        return transformed_conditioning, transformed_target
    
class Resize(torch.nn.Module):
    def __init__(
        self,
        size=(32, 32),
    ):
        super().__init__()
        self.size = size
        self.resize = VisionResize(
            size,
            antialias=True,
        )

    def forward(self, sample: Tensor) -> Tensor:
        conditioning, target = sample
        return self.resize(conditioning), self.resize(target)

class Translate(object):

    def __call__(self, sample):
        in_img, tgt_img = sample # (C, H, W)

        x_shift = torch.randint(in_img.shape[-2], (1,)).item()
        y_shift = torch.randint(in_img.shape[-1], (1,)).item()
        
        in_img = torch.roll(in_img, (x_shift, y_shift), dims=(-2, -1))
        tgt_img = torch.roll(tgt_img, (x_shift, y_shift), dims=(-2, -1))

        return in_img, tgt_img

class Flip(object):

    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, 'flipping is ambiguous for 1D scalars/vectors'

        self.axes = torch.randint(2, (self.ndim,), dtype=torch.bool)
        self.axes = torch.arange(self.ndim)[self.axes]

        in_img, tgt_img = sample

        if in_img.shape[0] == self.ndim:  # flip vector components
            in_img[self.axes] = - in_img[self.axes]

        shifted_axes = (1 + self.axes).tolist()
        in_img = torch.flip(in_img, shifted_axes)

        if tgt_img.shape[0] == self.ndim:  # flip vector components
            tgt_img[self.axes] = - tgt_img[self.axes]

        shifted_axes = (1 + self.axes).tolist()
        tgt_img = torch.flip(tgt_img, shifted_axes)

        return in_img, tgt_img

class Permutate(object):

    def __init__(self, ndim):
        self.axes = None
        self.ndim = ndim

    def __call__(self, sample):
        assert self.ndim > 1, 'permutation is not necessary for 1D fields'

        self.axes = torch.randperm(self.ndim)

        in_img, tgt_img = sample

        if in_img.shape[0] == self.ndim:  # permutate vector components
            in_img = in_img[self.axes]

        shifted_axes = [0] + (1 + self.axes).tolist()
        in_img = in_img.permute(shifted_axes)

        if tgt_img.shape[0] == self.ndim:  # permutate vector components
            tgt_img = tgt_img[self.axes]

        shifted_axes = [0] + (1 + self.axes).tolist()
        tgt_img = tgt_img.permute(shifted_axes)

        return in_img, tgt_img