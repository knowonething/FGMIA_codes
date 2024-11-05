import os

import PIL.Image
import torch
from loguru import logger as log
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, filename="celeba.txt"):
        self.root_dir = root_dir
        self.transform = transform
        if transform is None:
            self.transform = self.getDefaultTransforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)

        self.list_imgs = []

        if not os.path.exists(self.list_imgs_path):
            log.error("ERROR in readListFile {}", self.list_imgs_path)
            raise FileExistsError(self.list_imgs_path)

        with open(self.list_imgs_path, "r") as f:
            for index, context in enumerate(f):
                context = context.strip()
                one_line = context.split(" ")
                class_id = int(one_line[1])
                img_path = one_line[0]

                self.list_imgs.append(
                    {
                        'cid': class_id,
                        'img': img_path
                    }
                )
                if index % 10000 == 0:
                    log.info("[Dataset]: Processing to Line{}", index)

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        info = self.list_imgs[index]
        img_path = os.path.join(self.root_dir, "data/", info["img"])
        try:

            # log.info("infrx:{} Img:{}, label:{}", index, info['img'], info['cid'])
            img = PIL.Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, info['cid']
        except Exception as e:
            log.error("Error in read image:{}", e)
            log.error("Error in read image:{}", img_path)
            return None, None

    def getDefaultTransforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


class CelebAPairDataset(Dataset):
    def __init__(self, root_dir, transform=None, filename="celeba.txt"):
        self.root_dir = root_dir
        self.transform = transform
        if transform is None:
            self.transform = self.getDefaultTransforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)

        self.list_imgs = []

        if not os.path.exists(self.list_imgs_path):
            log.error("ERROR in readListFile {}", self.list_imgs_path)
            raise FileExistsError(self.list_imgs_path)

        with open(self.list_imgs_path, "r") as f:
            for index, context in enumerate(f):
                context = context.strip()
                one_line = context.split(" ")
                label = int(one_line[2])
                img1_path = one_line[0]
                img2_path = one_line[1]

                self.list_imgs.append(
                    {
                        'lb': label,
                        'img1': img1_path,
                        'img2': img2_path
                    }
                )
                if index % 10000 == 0:
                    log.info("[Dataset]: Processing to Line{}", index)

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        info = self.list_imgs[index]
        img1_path = os.path.join(self.root_dir, "data/", info["img1"])
        img2_path = os.path.join(self.root_dir, "data/", info["img2"])
        try:

            # log.info("infrx:{} Img:{}, label:{}", index, info['img'], info['cid'])
            img1 = PIL.Image.open(img1_path)
            img2 = PIL.Image.open(img2_path)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, info['lb']
        except Exception as e:
            log.error("Error in read image:{}", e)
            log.error("Error in read image:{}", info)
            return None, None

    def getDefaultTransforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


class CelebADatasetWithHiddenPth(Dataset):
    def __init__(self, root_dir, transform=None, filename="celeba.txt", hidden_base_path=""):
        self.root_dir = root_dir
        self.transform = transform
        self.hidden_base_path = hidden_base_path

        if transform is None:
            self.transform = self.getDefaultTransforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)

        self.list_imgs = []

        if not os.path.exists(self.list_imgs_path):
            log.error("ERROR in readListFile {}", self.list_imgs_path)
            raise FileExistsError(self.list_imgs_path)

        with open(self.list_imgs_path, "r") as f:
            for index, context in enumerate(f):
                context = context.strip()
                one_line = context.split(" ")
                class_id = int(one_line[1])
                img_path = one_line[0]
                hidden_path = img_path.replace(".png", ".pth")

                self.list_imgs.append(
                    {
                        'cid': class_id,
                        'img': img_path,
                        'hid': hidden_path
                    }
                )
                if index % 10000 == 0:
                    log.info("[Dataset]: Processing to Line{}", index)
            log.info("[Dataset]: End Process Dataset:{}", len(self.list_imgs))

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        info = self.list_imgs[index]
        img_path = os.path.join(self.root_dir, "data/", info["img"])
        hid_path = os.path.join(self.hidden_base_path, info["hid"])
        try:

            # log.info("infrx:{} Img:{}, label:{}", index, info['img'], info['cid'])
            img = PIL.Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            hid = torch.load(hid_path, map_location='cpu')
            return img, info['cid'], hid
        except Exception as e:
            log.error("Error in read image:{}", e)
            log.error("Error in read image:{}", img_path)
            return None, None, None

    def getDefaultTransforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


class CelebADatasetTopn(Dataset):
    def __init__(self, root_dir, transform=None, filename="celeba.txt"):
        self.root_dir = root_dir
        self.transform = transform
        if transform is None:
            self.transform = self.getDefaultTransforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)

        self.list_imgs = []

        if not os.path.exists(self.list_imgs_path):
            log.error("ERROR in readListFile {}", self.list_imgs_path)
            raise FileExistsError(self.list_imgs_path)

        with open(self.list_imgs_path, "r") as f:
            for index, context in enumerate(f):
                context = context.strip()
                one_line = context.split(" ")
                class_id = int(one_line[1])
                img_path = one_line[0]

                self.list_imgs.append(
                    {
                        'cid': class_id,
                        'img': img_path
                    }
                )
                if index % 10000 == 0:
                    log.info("[Dataset]: Processing to Line{}", index)

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        info = self.list_imgs[index]
        img_path = os.path.join(self.root_dir, "data/", info["img"])
        try:

            # log.info("infrx:{} Img:{}, label:{}", index, info['img'], info['cid'])
            img = PIL.Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, img_path
        except Exception as e:
            log.error("Error in read image:{}", e)
            log.error("Error in read image:{}", img_path)
            return None, None

    def getDefaultTransforms(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



if __name__ == "__main__":
    log.info("Hello, World!")
    root_dir = os.path.expanduser("~/dataset/celeba_prepared")
    dataset = CelebADataset(root_dir, filename="mini1000_1.txt")
    log.info("datalen :{}", len(dataset))
    image, label = dataset[0]
    # 给图像处理后到文件test.png
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
        transforms.ToPILImage(),
    ])

    # 将张量应用预处理转换
    image = preprocess(image)
    image.save("test.png")
