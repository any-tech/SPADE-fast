import torch
import os
import numpy as np


class Config:
    def __init__(self, args):
        self.args = args

        if self.args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:1')  # default

        # https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html#torchvision.models.Wide_ResNet50_2_Weights
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.SHAPE_MIDDLE = (args.load_size, args.load_size)  # (H, W)
        self.SHAPE_INPUT = (args.input_size, args.input_size)  # (H, W)

        # (H, W)
        self.pixel_crop = (
            int(abs(self.SHAPE_MIDDLE[0] - self.SHAPE_INPUT[0]) / 2),
            int(abs(self.SHAPE_MIDDLE[1] - self.SHAPE_INPUT[1]) / 2))

        # collect types of data
        types_data = [d for d in os.listdir(args.path_parent)
                      if os.path.isdir(os.path.join(args.path_parent, d))]
        self.types_data = np.sort(np.array(types_data))
        print('types_data =', self.types_data)
