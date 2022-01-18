import warnings
import albumentations as A
from albumentations.augmentations.transforms import PadIfNeeded

warnings.simplefilter("ignore")


# --------------------------------------------------------------------
# Helpful functions
# --------------------------------------------------------------------

def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")


# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------

post_transform = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

# crop 512
train_transform_1 = A.Compose([
    # A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
])

train_transform_2 = A.Compose([
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.PadIfNeeded(512, 512, p=1),
    A.RandomCrop(512, 512, p=1.),
    A.Flip(p=0.75),
])

train_transform_3 = A.Compose([
    A.Resize(height=224, width=224, p=1),
    A.RandomCrop(224, 224, p=1.),
    A.Flip(p=0.75),
])

train_transform_4 = A.Compose([
    # A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(256, 256, p=1.),
    A.Flip(p=0.75),
])

train_transform_5 = A.Compose([
    # A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(384, 384, p=1.),
    A.Flip(p=0.75),
])

# crop 1024
train_transform_7 = A.Compose([
    # A.RandomScale(scale_limit=0.3, p=0.5),
    A.RandomCrop(1024, 1024, p=1.),
    A.Flip(p=0.75),
])

valid_transform_1 = A.Compose([
])

test_transform_1 = A.Compose([
])

# resize 512
test_transform_2 = A.Compose([
    A.Resize(height=512, width=512, p=1)
])

