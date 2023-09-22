import os
import numpy as np
import torch

# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]])


class ConfigData:
    @classmethod
    def __init__(cls, args):
        # file reading related
        cls.num_cpu_max = args.num_cpu_max
        cls.path_parent = args.path_parent
        assert os.path.exists(cls.path_parent)

        # input format related
        cls.SHAPE_MIDDLE = (args.size_resize, args.size_resize)  # (H, W)
        cls.SHAPE_INPUT = (args.size_crop, args.size_crop)  # (H, W)
        cls.pixel_cut = (int((cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
                         int((cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))  # (H, W)

        # collect types of data
        types_data = [d for d in os.listdir(args.path_parent)
                      if os.path.isdir(os.path.join(args.path_parent, d))]
        cls.types_data = np.sort(np.array(types_data))


class ConfigFeat:
    def __init__(self, args):
        # adjsut to environment
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        # batch-size for feature extraction by ImageNet model
        self.batch_size = args.batch_size

        # input format related
        self.SHAPE_INPUT = (args.size_crop, args.size_crop)  # (H, W)

        # base network
        self.backbone = args.backbone
        self.weight = args.weight

        # layer specification
        self.layer_map = args.layer_map
        self.layer_vec = args.layer_vec

        # adjust to the network's learning policy and the data conditions
        self.MEAN = MEAN.to(self.device)
        self.STD = STD.to(self.device)


class ConfigSpade:
    def __init__(self, args):
        # number of nearest neighbor to get patch images
        self.k = args.k

        # input format related
        self.shape_stretch = (args.size_crop, args.size_crop)  # (H, W)

        # consideration for the outer edge
        self.pixel_outer_decay = args.pixel_outer_decay


class ConfigDraw:
    def __init__(self, args):
        # output detail or not (take a long time...)
        self.verbose = args.verbose

        # output filename related
        self.k = args.k

        # output path of figure
        self.path_result = args.path_result
