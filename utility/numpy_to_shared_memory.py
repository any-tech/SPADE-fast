import cv2
from config import Config
import numpy as np
from multiprocessing.sharedctypes import RawArray
from tqdm import tqdm
import multiprocessing as mp


def read_and_resize(file):
    img = cv2.imread(file)[..., ::-1]  # BGR2RGB
    img = cv2.resize(img, (Config.SHAPE_MIDDLE[1], Config.SHAPE_MIDDLE[0]), interpolation=cv2.INTER_AREA)
    img = img[
          Config.pixel_crop[0]:(Config.SHAPE_INPUT[0] + Config.pixel_crop[0]),
          Config.pixel_crop[1]:(Config.SHAPE_INPUT[1] + Config.pixel_crop[1])]
    SharedMemory.shared_array[np.where(SharedMemory.files == file)[0]] = img


class SharedMemory:
    shared_array = None
    files = None

    @classmethod
    def get_shared_memory_from_numpy(cls, config, files, img_array):
        SharedMemory.files = files
        SharedMemory.shared_array = img_array

        shape = (len(SharedMemory.files), config.SHAPE_INPUT[0], config.SHAPE_INPUT[1], 3)
        num_elm = shape[0] * shape[1] * shape[2] * shape[3]
        ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
        data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
        data.shape = shape
        SharedMemory.shared_array = data.view(np.uint8)

        # exec imread and imresize on multiprocess
        mp.set_start_method('fork', force=True)
        p = mp.Pool(min(mp.cpu_count(), config.args.num_cpu_max))

        for _ in tqdm(
                p.imap_unordered(read_and_resize, SharedMemory.files),
                total=len(SharedMemory.files),
                desc='read image for train'):
            pass

        p.close()

        return SharedMemory.shared_array
