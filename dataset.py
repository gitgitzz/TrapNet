import numpy as np
import os
import torch
import torchvision
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import utils


def sample_from_data(args, device, data_loader):
    """Sample real images and labels from data_loader.

    Args:
        args (argparse object)
        device (torch.device)
        data_loader (DataLoader)

    Returns:
        real, y

    """

    real, y = next(data_loader)
    real, y = real.to(device), y.to(device)

    return real, y


def sample_from_gen(args, device, num_classes, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        num_classes (int): for pseudo_y
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    pseudo_y = utils.sample_pseudo_labels(
        num_classes, args.batch_size, device
    )

    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


def generate_from_gen(args, device, pseudo_y, gen):
    """Sample fake images and labels from generator.

    Args:
        args (argparse object)
        device (torch.device)
        gen (nn.Module)

    Returns:
        fake, pseudo_y, z

    """

    z = utils.sample_z(
        args.batch_size, args.gen_dim_z, device, args.gen_distribution
    )
    fake = gen(z, pseudo_y)
    return fake, pseudo_y, z


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, args, root='', transform=None):
        super(FaceDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.images = []
        self.path = self.root
        self.images, self.labels = self._get_data_train_list()
        self.data_name = args.data_name

        # num_classes = len([lists for lists in os.listdir(
        #     self.path) if os.path.isdir(os.path.join(self.path, lists))])
        #
        # for idx in range(num_classes):
        #     class_path = os.path.join(self.path, str(idx))
        #     for _, _, files in os.walk(class_path):
        #         for img_name in files:
        #             image_path = os.path.join(class_path, img_name)
        #             image = Image.open(image_path)
        #             if args.data_name == 'facescrub':
        #                 if image.size != (64, 64):
        #                     image = image.resize((64, 64), Image.ANTIALIAS)
        #             self.images.append((image, idx))

    def _get_data_train_list(self):
        num_classes = len([lists for lists in os.listdir(self.path)
                           if os.path.isdir(os.path.join(self.path, lists))])
        images = []
        labels = []
        for idx in range(num_classes):
            class_path = os.path.join(self.path, str(idx))
            filename_list = os.listdir(class_path)
            for image_name in filename_list:
                image_path = os.path.join(class_path, image_name)
                images.append(image_path)
                labels.append(idx)
        return images, labels

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.data_name == 'facescrub':
            if image.size != (64, 64):
                image = image.resize((64, 64), Image.ANTIALIAS)
        if self.transform != None:
            image = self.transform(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L5-L15
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


# Copied from https://github.com/naoto0804/pytorch-AdaIN/blob/master/sampler.py#L18-L26
class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, file_path, transforms, mode="gan"):
        self.mode = mode
        self.processor = transforms
        self.data_root = root
        print("self.data_root:{}".format(self.data_root))
        self.name_list, self.label_list = self.get_list(file_path)
        self.num_img = len(self.name_list)
        print("example of self.name_list:", self.name_list[0])
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    def load_img(self, img_name):
        path = os.path.join(self.data_root, img_name)
        img = Image.open(path)
        img = img.convert('RGB')
        return img

    def __getitem__(self, index):
        img = self.processor(self.load_img(self.name_list[index]))
        if self.mode == "gan":
            return img
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return self.num_img


def init_dataset(args, mode="train", file_path=""):
    # dataset crop setting
    data_name = file_path.split("/")[-1].split("_")[0]
    if data_name == 'celeba':
        re_size = 64
        crop_size = 108
        data_root = os.path.join(args.data_root, "CelebA/img_align_celeba_png")
    elif data_name == 'ffhq':
        crop_size = 88
        re_size = 64
        data_root = os.path.join(args.data_root, "FFHQ/thumbnails128x128")
    elif data_name == 'facescrub':
        re_size = 64
        crop_size = 54
        data_root = os.path.join(args.data_root, "facescrub")
    elif data_name == 'vggface':
        re_size = 64
        data_root = os.path.join(args.data_root, "VGGFace/processed_faces")
    elif data_name == 'vggface2':
        re_size = 64
        data_root = os.path.join(args.data_root, "VGGFace2")
    else:
        raise Exception("Wrong Dataname!")

    if args.target_model == "FaceNet":
        re_size = 112

    transforms_list = []
    transforms_list.append(torchvision.transforms.ToTensor())
    if data_name in ['celeba', 'ffhq', 'facescrub']:
        transforms_list.append(torchvision.transforms.CenterCrop((crop_size, crop_size)))
    else:
        pass
    transforms_list.append(torchvision.transforms.Resize((re_size, re_size)))

    if 'train' in file_path:
        transforms_list.append(torchvision.transforms.RandomRotation(10))
        transforms_list.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))

    if data_name == 'celeba':
        pass
    elif data_name == 'ffhq':
        pass
    elif data_name == 'facescrub':
        pass
    elif data_name == "vggface":
        pass
        # transforms_list.append(torchvision.transforms.Normalize([129.1863, 104.7624, 93.5940], [1.0, 1.0, 1.0]))
    elif data_name == "vggface2":
        pass
        # transforms_list.append(torchvision.transforms.Normalize([131.0912, 103.8827, 91.4953], [1.0, 1.0, 1.0]))
    else:
        raise Exception("Invalid Dataset")

    transforms = torchvision.transforms.Compose(transforms_list)

    if mode == "train":
        dataset = ImageFolder(root=data_root, file_path=file_path, transforms=transforms, mode=mode)
    elif mode == "test":
        dataset = ImageFolder(root=data_root, file_path=file_path, transforms=transforms, mode=mode)
    elif mode == "gan_reclassified":
        dataset = ImageFolder(root=data_root, file_path=file_path, transforms=transforms, mode=mode)
    else:
        raise Exception("Invalid dataset mode:".format(mode))
    return dataset
