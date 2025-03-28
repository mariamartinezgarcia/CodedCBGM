import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate(
            [np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)],
            axis=2,
        )
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
      root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
      env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    def __init__(
        self, root="./data", env="train", transform=None, target_transform=None
    ):
        super(ColoredMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if env in ["train", "test"]:
            self.data_label_tuples = torch.load(
                os.path.join(self.root, "color_mnist", env) + ".pt"
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, [target, color_red, color_green] = self.data_label_tuples[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = torch.flatten(self.target_transform(target))

        return img, torch.cat((target, torch.tensor([color_red, color_green])))

    def __len__(self):
        return len(self.data_label_tuples)
        


class generate_data(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
      root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
      env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    def __init__(self, root="./data", env="train", transform=None, target_transform=None, confounded=False):
        super(generate_data, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if confounded:
            self.prepare_confounded_colored_mnist(env)
        else:
            self.prepare_colored_mnist(env)



    def prepare_colored_mnist(self, env):

        if env=='train':
            colored_mnist_dir = os.path.join(self.root, "color_mnist")
            if os.path.exists(
                os.path.join(colored_mnist_dir, "train.pt")
            ) and os.path.exists(os.path.join(colored_mnist_dir, "train.pt")):
                print("Colored MNIST dataset already exists")
                return

            print("Preparing Colored MNIST")
            train_mnist = datasets.mnist.MNIST("./data/mnist", train=True, download=True)

            train_set = []

            for idx, (im, label) in enumerate(train_mnist):
                if idx % 10000 == 0:
                    print(f"Converting image {idx}/{len(train_mnist)} in train mnist")
                im_array = np.array(im)
                if np.random.uniform() < 0.5:
                    color_red = 0
                    color_green = 1
                else:
                    color_red = 1
                    color_green = 0

                colored_arr = color_grayscale_arr(im_array, red=color_red)
                train_set.append(
                    (Image.fromarray(colored_arr), [label, color_red, color_green])
                )

            os.makedirs(colored_mnist_dir, exist_ok=True)
            torch.save(train_set, os.path.join(colored_mnist_dir, "train.pt"))

        if env=='test':
            colored_mnist_dir = os.path.join(self.root, "color_mnist")
            if os.path.exists(
                os.path.join(colored_mnist_dir, "test.pt")
            ) and os.path.exists(os.path.join(colored_mnist_dir, "test.pt")):
                print("Colored MNIST dataset already exists")
                return

            print("Preparing Colored MNIST")
            test_mnist = datasets.mnist.MNIST("./data/mnist", train=False, download=True)

            test_set = []

            for idx, (im, label) in enumerate(test_mnist):
                if idx % 10000 == 0:
                    print(f"Converting image {idx}/{len(test_mnist)} in test mnist")
                im_array = np.array(im)
                if np.random.uniform() < 0.5:
                    color_red = 0
                    color_green = 1
                else:
                    color_red = 1
                    color_green = 0

                colored_arr = color_grayscale_arr(im_array, red=color_red)
                test_set.append(
                    (Image.fromarray(colored_arr), [label, color_red, color_green])
                )

            os.makedirs(colored_mnist_dir, exist_ok=True)
            torch.save(test_set, os.path.join(colored_mnist_dir, "test.pt"))
    
    def prepare_confounded_colored_mnist(self, env):

        if env=='train':
            colored_mnist_dir = os.path.join(self.root, "color_mnist")
            if os.path.exists(
                os.path.join(colored_mnist_dir, "train.pt")
            ) and os.path.exists(os.path.join(colored_mnist_dir, "train.pt")):
                print("Confounded Colored MNIST dataset already exists")
                return

            print("Preparing Confounded Colored MNIST")
            train_mnist = datasets.mnist.MNIST("./data/mnist", train=True, download=True)

            train_set = []

            for idx, (im, label) in enumerate(train_mnist):
                if idx % 10000 == 0:
                    print(f"Converting image {idx}/{len(train_mnist)} in train mnist")
                im_array = np.array(im)
                if label % 2 == 0:
                    color_red = 0
                    color_green = 1
                else:
                    color_red = 1
                    color_green = 0

                colored_arr = color_grayscale_arr(im_array, red=color_red)
                train_set.append(
                    (Image.fromarray(colored_arr), [label, color_red, color_green])
                )

            os.makedirs(colored_mnist_dir, exist_ok=True)
            torch.save(train_set, os.path.join(colored_mnist_dir, "train.pt"))

        if env=='test':
            colored_mnist_dir = os.path.join(self.root, "color_mnist")
            if os.path.exists(
                os.path.join(colored_mnist_dir, "test.pt")
            ) and os.path.exists(os.path.join(colored_mnist_dir, "test.pt")):
                print("Colored MNIST dataset already exists")
                return

            print("Preparing Colored MNIST")
            test_mnist = datasets.mnist.MNIST("./data/mnist", train=False, download=True)

            test_set = []

            for idx, (im, label) in enumerate(test_mnist):
                if idx % 10000 == 0:
                    print(f"Converting image {idx}/{len(test_mnist)} in test mnist")
                im_array = np.array(im)
                if label % 2 == 0:
                    color_red = 0
                    color_green = 1
                else:
                    color_red = 1
                    color_green = 0

                colored_arr = color_grayscale_arr(im_array, red=color_red)
                test_set.append(
                    (Image.fromarray(colored_arr), [label, color_red, color_green])
                )

            os.makedirs(colored_mnist_dir, exist_ok=True)
            torch.save(test_set, os.path.join(colored_mnist_dir, "test.pt"))


def target_to_oh(target):
    NUM_CLASS = 10  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


def get_color_mnist(root, batch_size, istesting=False):


    if istesting:

        if not (os.path.isfile(root+"color_mnist/test.pt")):

            ###### if not exist create it
            generate_data(root=root, env='test')

        dl = torch.utils.data.DataLoader(
                ColoredMNIST(
                    root=root,
                    env="test",
                    target_transform=target_to_oh,
                    transform=transforms.Compose(
                    [
                        transforms.Resize(28),
                        transforms.ToTensor(),
                    ]
                    ),
                ),
                batch_size=batch_size,
                shuffle=True,
            )
    else:

        if not (os.path.isfile(root+"color_mnist/train.pt")):

            ###### if not exist create it
            generate_data(root=root)

        dl = torch.utils.data.DataLoader(
                ColoredMNIST(
                root=root,
                env="train",
                target_transform=target_to_oh,
                transform=transforms.Compose(
                    [
                        transforms.Resize(28),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    return dl



def get_confounded_color_mnist(root, batch_size, istesting=False):

    if istesting:

        if not (os.path.isfile(root+"confounded_color_mnist/test.pt")):

            print('generating dataset')
            ###### if not exist create it
            generate_data(root=root, confounded=True, env='test')

        dl = torch.utils.data.DataLoader(
                ColoredMNIST(
                root=root,
                env="test",
                target_transform=transforms.Compose([
                                 lambda x:torch.LongTensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x,10)]),
                transform=transforms.Compose(
                    [
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    else:

        if not (os.path.isfile(root+"confounded_color_mnist/train.pt")):

            ###### if not exist create it
            generate_data(root=root, confounded=True, env='train')

        dl = torch.utils.data.DataLoader(
                ColoredMNIST(
                root=root,
                env="train",
                target_transform=transforms.Compose([
                                 lambda x:torch.LongTensor([x]), # or just torch.tensor
                                 lambda x:F.one_hot(x,10)]),
                transform=transforms.Compose(
                    [
                        transforms.Resize(28),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

    return dl