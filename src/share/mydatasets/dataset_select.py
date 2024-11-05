import os

import torchvision
from PIL import Image
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import LFWPairs, ImageFolder

from src.share.mydatasets.datasets.CelebADataset import CelebADataset, CelebADatasetWithHiddenPth, CelebAPairDataset
from src.share.mydatasets.datasets.FFHQDataset import FFHQDataset, FFHQDatasetWithHidden
from src.share.mydatasets.datasets.VggfaceDataset import VGGFace2Dataset


def get_datasets(config, transform=None):
    if config.data_name == "celeba":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        dataset_base = CelebADataset(os.path.expanduser(config.data_rootPath), transform=transform,
                                     filename=config.data_fileName)
        if config.data_splitTrain > 0.0:
            train_dataset, test_dataset = random_split(dataset_base, [int(len(dataset_base) * config.data_splitTrain),
                                                                      len(dataset_base) - int(
                                                                          len(dataset_base) * config.data_splitTrain)])
            return train_dataset, test_dataset
        else:
            return dataset_base, None
    elif config.data_name == "celeba_hidden":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        dataset_base = CelebADatasetWithHiddenPth(
            os.path.expanduser(config.data_rootPath),
            transform=transform,
            filename=config.data_fileName,
            hidden_base_path=os.path.expanduser(config.data_hidden_base_path)
        )
        if config.data_splitTrain > 0.0:
            train_dataset, test_dataset = random_split(dataset_base, [int(len(dataset_base) * config.data_splitTrain),
                                                                      len(dataset_base) - int(
                                                                          len(dataset_base) * config.data_splitTrain)])
            return train_dataset, test_dataset
        else:
            return dataset_base, None

    elif config.data_name == "emnist":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.Lambda(lambda img: img.rotate(270) if isinstance(img, Image.Image) else img),
                transforms.RandomHorizontalFlip(p=1.),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        data_train = torchvision.datasets.EMNIST(root=os.path.expanduser(config.data_rootPath), download=False,
                                                 split=config.data_split,
                                                 train=True,
                                                 transform=transform)
        data_valid = torchvision.datasets.EMNIST(root=os.path.expanduser(config.data_rootPath), download=False,
                                                 split=config.data_split,
                                                 train=False,
                                                 transform=transform)
        return data_train, data_valid

    elif config.data_name == "cifar10":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        data_train = torchvision.datasets.CIFAR10(root=os.path.expanduser(config.data_rootPath), download=True,
                                                  train=True,
                                                  transform=transform)
        data_valid = torchvision.datasets.CIFAR10(root=os.path.expanduser(config.data_rootPath), download=True,
                                                  train=False,
                                                  transform=transform)
        return data_train, data_valid

    elif config.data_name == "cifar100":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        data_train = torchvision.datasets.CIFAR100(root=os.path.expanduser(config.data_rootPath), download=True,
                                                   train=True,
                                                   transform=transform)
        data_valid = torchvision.datasets.CIFAR100(root=os.path.expanduser(config.data_rootPath), download=True,
                                                   train=False,
                                                   transform=transform)
        return data_train, data_valid

    elif config.data_name == "ffhq":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        data_train = FFHQDataset(os.path.expanduser(config.data_rootPath), transform=transform)
        return data_train, None
    elif config.data_name == "vgg_new":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(config.data_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        data_train = ImageFolder(os.path.expanduser(config.data_rootPath), transform=transform)
        return data_train, None
    elif config.data_name == "ffhq_new":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(config.data_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        data_train = ImageFolder(os.path.expanduser(config.data_rootPath), transform=transform)
        return data_train, None
    elif config.data_name == "ffhq_hidden":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        dataset_base = FFHQDatasetWithHidden(
            os.path.expanduser(config.data_rootPath), transform=transform,
            hidden_base_path=os.path.expanduser(config.data_hidden_base_path)
        )
        if config.data_splitTrain > 0.0:
            train_dataset, test_dataset = random_split(dataset_base, [int(len(dataset_base) * config.data_splitTrain),
                                                                      len(dataset_base) - int(
                                                                          len(dataset_base) * config.data_splitTrain)])
            return train_dataset, test_dataset
        else:
            return dataset_base, None


    elif config.data_name == "vggface2":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        dataset_base = VGGFace2Dataset(os.path.expanduser(config.data_rootPath), transform=transform,
                                       filename=config.data_fileName)
        if config.data_splitTrain > 0.0:
            train_dataset, test_dataset = random_split(dataset_base, [int(len(dataset_base) * config.data_splitTrain),
                                                                      len(dataset_base) - int(
                                                                          len(dataset_base) * config.data_splitTrain)])
            return train_dataset, test_dataset
        else:
            return dataset_base, None
    elif config.data_name == "lfw":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        train_dataset = LFWPairs(root=os.path.expanduser(config.data_rootPath), split="train",
                                 image_set=config.data_image_set, transform=transform,
                                 download=False)

        test_dataset = LFWPairs(root=os.path.expanduser(config.data_rootPath), split="test",
                                image_set=config.data_image_set, transform=transform,
                                download=False)
        return train_dataset, test_dataset
    elif config.data_name == "celeba_pair":
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((config.data_size, config.data_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        dataset_base = CelebAPairDataset(os.path.expanduser(config.data_rootPath), transform=transform,
                                         filename=config.data_fileName)

        return dataset_base, dataset_base

    # elif config.data_name == "casia_webface":
    #     if transform is None:
    #         transform = transforms.Compose([
    #             transforms.Resize((config.data_size, config.data_size)),
    #             transforms.RandomRotation(10),
    #             transforms.RandomHorizontalFlip(p=1.),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #         ])
    #     dataset_base = CASIAWebFaceDataset(os.path.expanduser(config.data_rootPath), transform=transform,
    #                                        filename=config.data_fileName)
    #     if config.data_splitTrain > 0.0:
    #         train_dataset, test_dataset = random_split(dataset_base, [int(len(dataset_base) * config.data_splitTrain),
    #                                                                   len(dataset_base) - int(
    #                                                                       len(dataset_base) * config.data_splitTrain)])
    #         return train_dataset, test_dataset
    #     else:
    #         return dataset_base, None
    else:
        raise NotImplementedError(f"Unknown dataset {config.data_name}")


if __name__ == "__main__":
    class Config:
        data_name = "casia_webface"
        data_rootPath = "~/dataset/casia_webface"
        data_fileName = "mini1000_1.txt"
        data_splitTrain = 0.8
        data_size = 256


    conf = Config()

    train_set, val_set = get_datasets(config=conf)

    print(len(train_set))
    print(train_set[0][0].shape)

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化
        transforms.ToPILImage(),
    ])
    img = preprocess(train_set[0][0])
    img.save("test.png")
