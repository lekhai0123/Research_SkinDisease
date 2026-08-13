from PIL import Image, ImageEnhance
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FixedCenterZoom:
    def __init__(self, scale=1.2):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        left = (nw - w) // 2
        top = (nh - h) // 2
        return img.crop((left, top, left + w, top + h))


class FixedRotate:
    def __init__(self, angle=20):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, self.angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)


class FixedBrightness:
    def __init__(self, factor=1.2):
        self.factor = factor

    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(self.factor)


class FixedShear:
    def __init__(self, shear=12):
        self.shear = shear

    def __call__(self, img):
        return TF.affine(
            img,
            angle=0,
            translate=[0, 0],
            scale=1.0,
            shear=[self.shear, 0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )


def build_eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


def build_augmentation_transforms(img_size=224):
    """Named gallery of fixed augmentations used to build the balanced training set.

    Each entry is composed with the same resize/normalize tail as the eval transform,
    matching the augmentation protocol described in the paper (rotation, shear, center
    zoom, brightness, horizontal flip, vertical flip) plus the unmodified original.
    """
    eval_tail = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]

    return {
        "orig": transforms.Compose(eval_tail),
        "zoom": transforms.Compose([FixedCenterZoom(1.2), *eval_tail]),
        "rot": transforms.Compose([FixedRotate(20), *eval_tail]),
        "bright": transforms.Compose([FixedBrightness(1.2), *eval_tail]),
        "shear": transforms.Compose([FixedShear(12), *eval_tail]),
        "vflip": transforms.Compose([transforms.Lambda(lambda x: TF.vflip(x)), *eval_tail]),
        "hflip": transforms.Compose([transforms.Lambda(lambda x: TF.hflip(x)), *eval_tail]),
    }
