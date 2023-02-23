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
    gts_test = {}

    def __init__(self, args, config, type_data):
        super().__init__()
        self.args = args
        self.type_data = type_data
        self.config = config
        self.num_channel = 3

        MVTecDataset.image_train = None
        MVTecDataset.image_test = {}
        MVTecDataset.gts_test = {}

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

        # for ground truth data
        for type_test in self.types_test:
            # create memory shared variable
            if type_test == 'good':
                shape = (len(self.files_test[type_test]), self.config.SHAPE_INPUT[0], self.config.SHAPE_INPUT[1])
                MVTecDataset.gts_test[type_test] = np.zeros(shape, dtype=np.uint8)
            else:
                MVTecDataset.gts_test[type_test] = None
                MVTecDataset.gts_test[type_test] = SharedMemory.get_shared_memory_from_numpy(
                    self.config,
                    self.files_test[type_test],
                    MVTecDataset.gts_test[type_test],
                    is_ground_truth=True
                )


