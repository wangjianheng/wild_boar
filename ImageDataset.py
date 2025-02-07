import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):

    def __init__(self, root, width, height):
        self.root = root
        self.width = width
        self.height = height
        self.enumerates = []
        self.images = self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        name, label = self.images[i]

        image = self.pretreatment(
            os.path.join(self.root, name)
        )

        image = image.unsqueeze(0)

        return image, label

    def load_images(self):
        if not os.path.isdir(self.root):
            raise ValueError('load path err')

        file_list = os.listdir(self.root)
        count = len(file_list)

        images = []

        for file in file_list:
            name, _ = os.path.splitext(file)

            names = name.split('-')

            label = int(names[-1])

            if label not in self.enumerates:
                self.enumerates.append(label)

            images.append([file, label])

        print('load images finish:%d' % count)
        return images

    def pretreatment(self, path):
        image = Image.open(path)

        size = (self.width, self.height)

        image = image.convert('L').resize(size)

        image = np.array(image)

        return torch.from_numpy(image).float()
