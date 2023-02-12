import torch
import os
import numpy as np
import PIL
import torch
from torchvision import transforms
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray


class MVTecDataset(torch.utils.data.Dataset):

    def __init__(self, args, config, type_data, is_train=True):
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.type_data = type_data
        self.config = config
        self.num_channel = 3

        self.transform_img = [
            transforms.Resize(self.config.SHAPE_MIDDLE),
            transforms.CenterCrop(self.config.SHAPE_INPUT),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.MEAN, std=self.config.STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        if self.is_train:
            path = os.path.join(self.args.path_parent, self.type_data, 'train/good')
            files = [os.path.join(path, f) for f in os.listdir(path)
                           if (os.path.isfile(os.path.join(path, f)) & ('.png' in f))]
            self.files = np.sort(np.array(files))
        else:
            self.files_test = {}
            types_test = os.listdir(os.path.join(args.path_parent, type_data, 'test'))
            types_test = np.array(sorted(types_test))
            for type_test in types_test:
                path_test = os.path.join(args.path_parent, type_data, 'test', type_test)
                self.files[type_test] = [os.path.join(path_test, f)
                                         for f in os.listdir(path_test)
                                         if (os.path.isfile(os.path.join(path_test, f)) & ('.png' in f))]
                self.files[type_test] = np.sort(np.array(self.files[type_test]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.is_train:
            file = self.files[idx]
            image = PIL.Image.open(file).convert("RGB")
            image = self.transform_img(image)

            num_elm = image.shape[0] * image.shape[1] * image.shape[2]
            ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
            data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
            data.shape = image[None].shape
            shared_mem = data.view(np.uint8)
            shared_mem[0] = image

        else:
            pass

        return shared_mem
