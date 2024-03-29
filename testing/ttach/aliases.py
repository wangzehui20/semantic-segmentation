from .base import Compose
from . import transforms as tta


def flip_transform():
    return Compose([tta.HorizontalFlip(), tta.VerticalFlip()])


def hflip_transform():
    return Compose([tta.HorizontalFlip()])


def vlip_transform():
    return Compose([tta.VerticalFlip()])


def d4_transform():
    return Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

def d21_transform():
    return Compose(
        [
            tta.VerticalFlip(),
            tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180])
        ]
    )

def d5_transform():
    return Compose(
        [
            # tta.HorizontalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

def multiscale_transform(scales, interpolation="nearest"):
    return Compose([tta.Scale(scales, interpolation=interpolation)])


def five_crop_transform(crop_height, crop_width):
    return Compose([tta.FiveCrops(crop_height, crop_width)])


def ten_crop_transform(crop_height, crop_width):
    return Compose([tta.HorizontalFlip(), tta.FiveCrops(crop_height, crop_width)])
