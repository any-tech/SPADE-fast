import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchinfo import summary


class FeatExtract:
    def __init__(self, cfg_feat):
        self.device = cfg_feat.device
        self.batch_size = cfg_feat.batch_size
        self.shape_input = cfg_feat.SHAPE_INPUT
        self.layer_map = cfg_feat.layer_map
        self.layer_vec = cfg_feat.layer_vec
        self.MEAN = cfg_feat.MEAN
        self.STD = cfg_feat.STD

        code = 'self.backbone = %s(weights=%s)' % (cfg_feat.backbone, cfg_feat.weight)
        exec(code)
        self.backbone.eval()
        self.backbone.to(self.device)
        summary(self.backbone, input_size=(1, 3, *self.shape_input))

        self.features = []
        for layer_map in self.layer_map:
            code = 'self.backbone.%s.register_forward_hook(self.hook)' % layer_map
            exec(code)
        code = 'self.backbone.%s.register_forward_hook(self.hook)' % self.layer_vec
        exec(code)

    def hook(self, module, input, output):
        self.features.append(output.detach().cpu().numpy())

    def survey_depth(self):
        # feature extract for train
        x = torch.zeros(1, 3, self.shape_input[0], self.shape_input[1])  # RGB
        x = x.to(self.device)
        self.features = []
        with torch.no_grad():
            _ = self.backbone(x)

        depth = []
        for i_layer_map in range(len(self.layer_map)):
            depth.append(self.features[i_layer_map].shape[1])
        depth.append(self.features[-1].squeeze(-1).squeeze(-1).shape[1])
        return depth

    def normalize(self, input):
        x = torch.from_numpy(input.astype(np.float32))
        x = x.to(self.device)
        x = x / 255
        x = x - self.MEAN
        x = x / self.STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        return x

    def extract(self, imgs, is_train=True):  # is_train=False->dict, is_train=True->np.array
        if is_train:
            # feature extract for train
            x_batch = []
            self.features = []
            for i, img in tqdm(enumerate(imgs),
                               desc='feature extract for train (case:good)'):
                x = self.normalize(img)
                x_batch.append(x)

                if ((len(x_batch) == self.batch_size) |
                    (i == (len(imgs) - 1))):
                    with torch.no_grad():
                        _ = self.backbone(torch.vstack(x_batch))
                    x_batch = []

            feat_map = []
            num_layer = len(self.layer_map) + 1
            for i_layer_map in range(len(self.layer_map)):
                # (warning) There is a limit of np.vstack...
                feat_map.append(np.vstack(self.features[i_layer_map::num_layer]))
            feat_vec = np.vstack(self.features[len(self.layer_map)::num_layer])
            feat_vec = feat_vec.squeeze(-1).squeeze(-1)
        else:
            # feature extract for test
            feat_map = {}
            feat_vec = {}
            for type_test in imgs.keys():
                x_batch = []
                self.features = []
                for i, img in tqdm(enumerate(imgs[type_test]),
                                   desc='feature extract for test (case:%s)' % type_test):
                    x = self.normalize(img)
                    x_batch.append(x)

                    if ((len(x_batch) == self.batch_size) |
                        (i == (len(imgs[type_test]) - 1))):
                        with torch.no_grad():
                            _ = self.backbone(torch.vstack(x_batch))
                        x_batch = []

                feat_map[type_test] = []
                num_layer = len(self.layer_map) + 1
                for i_layer_map in range(len(self.layer_map)):
                    feat_map[type_test].append(np.vstack(self.features[i_layer_map::num_layer]))
                feat_vec[type_test] = np.vstack(self.features[len(self.layer_map)::num_layer])
                feat_vec[type_test] = feat_vec[type_test].squeeze(-1).squeeze(-1)

        return feat_map, feat_vec
