import torch
import os
import numpy as np
import PIL
import torch
from torchvision import transforms
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
import cv2
from config import Config
from tqdm import tqdm
from utility.numpy_to_shared_memory import SharedMemory
from torch.utils.data.sampler import BatchSampler


class MVTecDataset(torch.utils.data.Dataset):
    image_train = None
    image_test = {}

    def __init__(self, args, config, type_data):
        super().__init__()
        self.args = args
        self.type_data = type_data
        self.config = config
        self.num_channel = 3

        # for train data
        path = os.path.join(self.args.path_parent, self.type_data, 'train/good')
        files = [os.path.join(path, f) for f in os.listdir(path)
                       if (os.path.isfile(os.path.join(path, f)) & ('.png' in f))]

        self.files = np.sort(np.array(files))
        MVTecDataset.image_train = SharedMemory.get_shared_memory_from_numpy(
            self.config,
            self.files,
            MVTecDataset.image_train)

        # for test data
        self.files_test = {}
        self.types_test = os.listdir(os.path.join(args.path_parent, type_data, 'test'))
        self.types_test = np.array(sorted(self.types_test))
        for type_test in self.types_test:
            path_test = os.path.join(args.path_parent, type_data, 'test', type_test)
            self.files_test[type_test] = [
                os.path.join(path_test, f) for f in os.listdir(path_test)
                if (os.path.isfile(os.path.join(path_test, f)) & ('.png' in f))
            ]
            self.files_test[type_test] = np.sort(np.array(self.files_test[type_test]))

            MVTecDataset.image_test[type_test] = None
            MVTecDataset.image_test[type_test] = SharedMemory.get_shared_memory_from_numpy(
                self.config,
                self.files_test[type_test],
                MVTecDataset.image_test[type_test])

    def normalize(self, input):
        x = torch.from_numpy(input.astype(np.float32)).to(self.config.device)
        x = x / 255
        x = x - self.config.MEAN
        x = x / self.config.STD
        x = x.permute(2, 0, 1)
        return x

    # def __len__(self):
    #     if self.is_train:
    #         return len(self.files)
    #     else:
    #         return len(self.files_test[self.types_test[0]])
    #
    # def __getitem__(self, idx):
    #     if self.is_train:
    #         img = MVTecDataset.image_train[idx]
    #     else:
    #         img = MVTecDataset.image_test[idx]
    #
    #     x = self.normalize(img)
    #
    #     return x
