import random

from PIL import Image
from torchvision import transforms


class CenterCropWithRandomOffset(object):
    def __init__(self, size, offset_range=10):
        self.size = size
        self.offset_range = offset_range

    def __call__(self, image):
        # Get the original image size
        c, h, w = image.shape

        # Calculate the center coordinates
        center_h = h // 2
        center_w = w // 2

        # Generate random offsets
        offset_h = random.randint(-self.offset_range, self.offset_range)
        offset_w = random.randint(-self.offset_range, self.offset_range)

        # Calculate the crop coordinates
        start_h = center_h + offset_h - self.size // 2
        start_w = center_w + offset_w - self.size // 2
        end_h = start_h + self.size
        end_w = start_w + self.size

        # Crop the image
        cropped_image = image[:, start_h:end_h, start_w:end_w]

        return cropped_image

class CenterCropWithRec(object):
    def __init__(self, size, offset_range=10):
        self.size = size
        self.offset_range = offset_range

    def __call__(self, image):
        # Get the original image size
        c, h, w = image.shape

        # Calculate the center coordinates
        center_h = h // 2
        center_w = w // 2

        # Generate random offsets
        offset_h = random.randint(-self.offset_range, self.offset_range)
        offset_w = random.randint(-self.offset_range, self.offset_range)

        # Calculate the crop coordinates
        start_h = center_h + offset_h - self.size[0] // 2
        start_w = center_w + offset_w - self.size[1] // 2
        end_h = start_h + self.size[0]
        end_w = start_w + self.size[0]

        # Crop the image
        cropped_image = image[:, start_h:end_h, start_w:end_w]

        return cropped_image



def get_transfroms(config):
    if config.data_crop:
        if config.data_name == "celeba" or config.data_name == "celeba_hidden" or config.data_name == "ffhq" or config.data_name == "ffhq_hidden" or config.data_name == "lfw" or config.data_name == "celeba_pair":
            trans = transforms.Compose([
                transforms.ToTensor(),
                CenterCropWithRandomOffset(config.data_crop_size, config.data_crop_offset),
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=config.data_h_flip_p),
                transforms.Resize((config.data_size, config.data_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            trans = None
    else:
        if config.data_crop_rec:
            trans = transforms.Compose([
                transforms.ToTensor(),
                CenterCropWithRec(config.data_crop_size, config.data_crop_offset),
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=config.data_h_flip_p),
                transforms.Resize(config.data_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            trans = None
    if config.data_name == "emnist" and (config.data_split == "letters" or config.data_split == "balanced"):
        trans = transforms.Compose([
            transforms.Resize((config.data_size, config.data_size)),
            transforms.Lambda(lambda img: img.rotate(270) if isinstance(img, Image.Image) else img),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((-20, 20)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    return trans
