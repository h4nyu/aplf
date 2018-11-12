import numpy as np
import albumentations as A
from torchvision.transforms import (
    ToTensor
)


class Augment(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        out = self.transform(image=image)['image'].reshape(*image.shape).astype(np.float32)
        return ToTensor()(out)


def batch_aug(aug, batch, ch=3):
    return pipe(
        batch,
        map(lambda x: [aug(x[0:ch, :, :]), aug(x[ch:2*ch, :, :])]),
        map(lambda x: torch.cat(x, dim=0)),
        list,
        torch.stack
    )
