import torch
import os
import numpy as np


class Config:
    @classmethod
    def __init__(cls, args):
        cls.args = args

        if cls.args.cpu:
            cls.device = torch.device('cpu')
        else:
            cls.device = torch.device('cuda:1')  # default

        # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html#torchvision.models.Wide_ResNet50_2_Weights
        cls.MEAN = [0.485, 0.456, 0.406]
        cls.STD = [0.229, 0.224, 0.225]
        cls.SHAPE_MIDDLE = (args.load_size, args.load_size)  # (H, W)
        cls.SHAPE_INPUT = (args.input_size, args.input_size)  # (H, W)

        # (H, W)
        cls.pixel_crop = (
            int(abs(cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
            int(abs(cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))

        # collect types of data
        types_data = [d for d in os.listdir(args.path_parent)
                      if os.path.isdir(os.path.join(args.path_parent, d))]
        cls.types_data = np.sort(np.array(types_data))
        print('types_data =', cls.types_data)
