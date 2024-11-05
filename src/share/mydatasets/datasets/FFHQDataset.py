import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from loguru import logger as log
from torchvision import transforms

from src.share.mydatasets.datasets.CelebADataset import CelebADataset


class FFHQDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.images = []
        self.load_data()

    def load_data(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filepath = self.images[idx]
        label = int(os.path.basename(filepath).replace(".png", "").replace(".jpg", ""))

        image = Image.open(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class FFHQDatasetWithHidden(Dataset):
    def __init__(self, path, transform=None, hidden_base_path=""):
        self.path = path
        self.transform = transform
        self.images = []
        self.hidden_base_path = hidden_base_path
        self.load_data()

    def load_data(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filepath = self.images[idx]
        label = int(os.path.basename(filepath).replace(".png", "").replace(".jpg", ""))
        hidden_path = os.path.join(self.hidden_base_path, os.path.basename(filepath).replace(".png", ".pth").replace(".jpg", ".pth"))
        image = Image.open(filepath)
        hidden = torch.load(hidden_path, map_location='cpu')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, hidden


if __name__ == "__main__":
    log.info("Hello, World!")
    root_dir = os.path.expanduser("~/dataset/ffhq128")
    dataset = FFHQDatasetWithHidden(root_dir,
                                    hidden_base_path="/home/miao/dataset/ffhq_arcface/glint360k_cosface_r18_fp16_0.1_bin")
    log.info("datalen :{}", len(dataset))
    image, label, hidden = dataset[1]
    # 给图像处理后到文件test.png
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
        transforms.ToPILImage(),
    ])
    log.info("hidden.shape:{}", hidden.shape)
    # log.info("image.shape:{}", image.shape)
    # 将张量应用预处理转换
    # image = preprocess(image)
    # image.save("test.png")
