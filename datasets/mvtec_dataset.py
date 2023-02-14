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


def read_and_resize(file):
    img = cv2.imread(file)[..., ::-1]  # BGR2RGB
    img = cv2.resize(img, (Config.SHAPE_MIDDLE[1], Config.SHAPE_MIDDLE[0]), interpolation=cv2.INTER_AREA)
    img = img[
          Config.pixel_crop[0]:(Config.SHAPE_INPUT[0] + Config.pixel_crop[0]),
          Config.pixel_crop[1]:(Config.SHAPE_INPUT[1] + Config.pixel_crop[1])]
    MVTecDataset.imgs_share_array[np.where(MVTecDataset.files == file)[0]] = img


class MVTecDataset(torch.utils.data.Dataset):
    imgs_share_array = None
    files = None

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

            MVTecDataset.files = np.sort(np.array(files))
            self.files = np.sort(np.array(files))

            shape = (len(self.files), self.config.SHAPE_INPUT[0], self.config.SHAPE_INPUT[1], 3)
            num_elm = shape[0] * shape[1] * shape[2] * shape[3]
            ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
            data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
            data.shape = shape
            MVTecDataset.imgs_share_array = data.view(np.uint8)

            # exec imread and imresize on multiprocess
            mp.set_start_method('fork', force=True)
            p = mp.Pool(min(mp.cpu_count(), config.args.num_cpu_max))

            for _ in tqdm(
                    p.imap_unordered(read_and_resize, MVTecDataset.files),
                    total=len(MVTecDataset.files),
                    desc='read image for train'):
                pass
            p.close()

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
            ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.float32))
            data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
            data.shape = image[None].shape
            shared_mem = data.view(np.float32)
            shared_mem[0] = image
            shared_mem = shared_mem.squeeze(0)

        else:
            pass

        return shared_mem
