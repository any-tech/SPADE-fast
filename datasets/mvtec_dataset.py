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


class MVTecDataset:
    @classmethod
    def __init__(cls, args, type_data):
        cls.args = args
        cls.type_data = type_data
        cls.num_channel = 3

        cls.image_train = None
        cls.image_test = {}
        cls.gts_test = {}

        # for train data
        print('***** train data *****')
        path = os.path.join(cls.args.path_parent, cls.type_data, 'train/good')
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if (os.path.isfile(os.path.join(path, f)) & ('.png' in f))]

        cls.files = np.sort(np.array(files))
        cls.image_train = SharedMemory.get_shared_memory_from_numpy(cls.files, cls.image_train)

        # for test data
        print('***** test data *****')
        cls.files_test = {}
        cls.types_test = os.listdir(os.path.join(args.path_parent, type_data, 'test'))
        cls.types_test = np.array(sorted(cls.types_test))
        for type_test in cls.types_test:
            path_test = os.path.join(args.path_parent, type_data, 'test', type_test)
            cls.files_test[type_test] = [
                os.path.join(path_test, f) for f in os.listdir(path_test)
                if (os.path.isfile(os.path.join(path_test, f)) & ('.png' in f))
            ]
            cls.files_test[type_test] = np.sort(np.array(cls.files_test[type_test]))

            cls.image_test[type_test] = None
            cls.image_test[type_test] = SharedMemory.get_shared_memory_from_numpy(
                cls.files_test[type_test],
                cls.image_test[type_test]
            )

        # for ground truth data
        print('***** ground truth data *****')
        for type_test in cls.types_test:
            # create memory shared variable
            if type_test == 'good':
                shape = (len(cls.files_test[type_test]), Config.SHAPE_INPUT[0], Config.SHAPE_INPUT[1])
                cls.gts_test[type_test] = np.zeros(shape, dtype=np.uint8)
            else:
                cls.gts_test[type_test] = None
                cls.gts_test[type_test] = SharedMemory.get_shared_memory_from_numpy(
                    cls.files_test[type_test],
                    cls.gts_test[type_test],
                    is_ground_truth=True
                )


