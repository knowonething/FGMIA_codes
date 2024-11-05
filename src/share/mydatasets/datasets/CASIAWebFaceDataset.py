import os

import PIL.Image
from torch.utils.data import Dataset
from torchvision import transforms


class CASIAWebFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, filename="all.txt"):
        self.root_dir = root_dir
        self.transform = transform
        if transform is None:
            self.transform = self.getDefaultTransforms()
        self.list_imgs_path = os.path.join(self.root_dir, filename)

        self.list_imgs = []

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

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        info = self.list_imgs[index]
        img_path = os.path.join(self.root_dir, info["img"])
        img = PIL.Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, info['cid']

    def getDefaultTransforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
