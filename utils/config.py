import os
import numpy as np
import torch

# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]])

class ConfigData:
    @classmethod
    def __init__(cls, args):
        # input format related
        cls.SHAPE_MIDDLE = (args.resize_size, args.resize_size)  # (H, W)
        cls.SHAPE_INPUT = (args.crop_size, args.crop_size)  # (H, W)
        cls.pixel_cut = (int((cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
                         int((cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))  # (H, W)

        # file reading related
        cls.num_cpu_max = args.num_cpu_max
        cls.path_parent = args.path_parent
        assert os.path.exists(cls.path_parent)

        # collect types of data
        types_data = [d for d in os.listdir(args.path_parent)
                      if os.path.isdir(os.path.join(args.path_parent, d))]
        cls.types_data = np.sort(np.array(types_data))


class ConfigFeat:
    @classmethod
    def __init__(cls, args):
        # input format related
        cls.SHAPE_INPUT = (args.crop_size, args.crop_size)  # (H, W)

        # base network
        cls.backbone = args.backbone

        # adjsut to environment
        cls.batch_size = args.batch_size
        if args.cpu:
            cls.device = torch.device('cpu')
        else:
            cls.device = torch.device('cuda:0')

        # adjust to the network's learning policy and the data conditions
        cls.MEAN = MEAN.to(cls.device)
        cls.STD = STD.to(cls.device)


class ConfigSpade:
    @classmethod
    def __init__(cls, args):
        # number of nearest neighbor to get patch images
        cls.k = args.k
        # input format related
        cls.shape_stretch = (args.crop_size, args.crop_size)  # (H, W)


class ConfigDraw:
    @classmethod
    def __init__(cls, args):
        # output path of figure
        cls.path_result = args.path_result
        # output detail or not (take a long time...)
        cls.verbose = args.verbose
        # number of nearest neighbor to get patch images
        cls.k = args.k
