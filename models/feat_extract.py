import numpy as np
from tqdm import tqdm
import torch
import torchvision.models as models
from torchinfo import summary


class FeatExtract:
    def __init__(self, cfg_feat):
        self.shape_input = cfg_feat.SHAPE_INPUT
        self.device = cfg_feat.device
        self.MEAN = cfg_feat.MEAN
        self.STD = cfg_feat.STD
        self.batch_size = cfg_feat.batch_size

        if (cfg_feat.backbone == 'wide_resnet50_2'):
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            self.backbone = models.wide_resnet50_2(weights=weights)
        else:
            assert False  # not prepared...

        self.backbone.eval()
        self.backbone.to(cfg_feat.device)
        summary(self.backbone, input_size=(1, 3, cfg_feat.SHAPE_INPUT[0], 
                                                 cfg_feat.SHAPE_INPUT[1]))

        self.features = []
        self.backbone.layer1[-1].register_forward_hook(self.hook)
        self.backbone.layer2[-1].register_forward_hook(self.hook)
        self.backbone.layer3[-1].register_forward_hook(self.hook)
        self.backbone.avgpool.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.features.append(output.detach().cpu().numpy())

    def survey_depth(self):
        # feature extract for train
        x = torch.zeros(1, 3, self.shape_input[0], self.shape_input[1])
        x = x.to(self.device)
        self.features = []
        with torch.no_grad():
            _ = self.backbone(x)
        feat_map_l1 = self.features[0]
        feat_map_l2 = self.features[1]
        feat_map_l3 = self.features[2]
        feat_vec_lf = self.features[3].squeeze(-1).squeeze(-1)
        depth = [feat_map_l1.shape[1], feat_map_l2.shape[1],
                 feat_map_l3.shape[1], feat_vec_lf.shape[1]]
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

            feat_map_l1 = np.vstack(self.features[0::4])
            feat_map_l2 = np.vstack(self.features[1::4])
            feat_map_l3 = np.vstack(self.features[2::4])
            feat_vec_lf = np.vstack(self.features[3::4]).squeeze(-1).squeeze(-1)
        else:
            # feature extract for test
            feat_map_l1 = {}
            feat_map_l2 = {}
            feat_map_l3 = {}
            feat_vec_lf = {}
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

                feat_map_l1[type_test] = np.vstack(self.features[0::4])
                feat_map_l2[type_test] = np.vstack(self.features[1::4])
                feat_map_l3[type_test] = np.vstack(self.features[2::4])
                feat_vec_lf[type_test] = np.vstack(self.features[3::4]).squeeze(-1).squeeze(-1)

        feat_map = [feat_map_l1, feat_map_l2, feat_map_l3]
        return feat_map, feat_vec_lf
    