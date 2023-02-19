import torch
from torchinfo import summary
from torch import nn
from tqdm import tqdm
import numpy as np
from datasets.mvtec_dataset import MVTecDataset
import faiss


class Spade(nn.Module):
    def __init__(self, args, config, dataset):
        super(Spade, self).__init__()
        self.args = args
        self.config = config

        if self.args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:1')  # default

        self.backbone = torch.hub.load('pytorch/vision:v0.12.0', 'wide_resnet50_2', pretrained=True)
        self.backbone.eval()
        self.backbone.to(self.device)
        summary(self.backbone, input_size=(1, 3, self.args.input_size, self.args.input_size))

        self.backbone.layer1[-1].register_forward_hook(self.hook)
        self.backbone.layer2[-1].register_forward_hook(self.hook)
        self.backbone.layer3[-1].register_forward_hook(self.hook)
        self.backbone.avgpool.register_forward_hook(self.hook)

        self.dataset = dataset
        self.features = []
        self.normal_patches = None

        self.f1_train = None
        self.f2_train = None
        self.f3_train = None
        self.fl_train = None

        self.f1_test = {}
        self.f2_test = {}
        self.f3_test = {}
        self.fl_test = {}

    def hook(self, module, input, output):
        self.features.append(output.detach().cpu().numpy())

    def forward(self, img):
        # self.features.clear()
        self.backbone(img)
        return self.features

    def normalize(self, input):
        x = torch.from_numpy(input.astype(np.float32)).to(self.config.device)
        x = x / 255
        x = x - self.config.MEAN
        x = x / self.config.STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        return x

    def create_normal_features(self):
        # feature extract for train
        x_batch = []
        self.features = []
        for i, img in tqdm(enumerate(MVTecDataset.image_train), desc='feature extract for train'):
            x = self.normalize(img)
            x_batch.append(x)

            if (len(x_batch) == self.args.batch_size) | (i == (len(MVTecDataset.image_train) - 1)):
                with torch.no_grad():
                    _ = self(torch.vstack(x_batch))
                x_batch = []

        self.f1_train = np.vstack(self.features[0::4])
        self.f2_train = np.vstack(self.features[1::4])
        self.f3_train = np.vstack(self.features[2::4])
        self.fl_train = np.vstack(self.features[3::4]).squeeze(-1).squeeze(-1)

    def create_test_features(self):
        # feature extract for test
        for type_test in self.dataset.types_test:
            x_batch = []
            self.features = []
            for i, img in tqdm(enumerate(MVTecDataset.image_test[type_test]), desc='feature extract for test (case:%s)' % type_test):
                x = self.normalize(img)
                x_batch.append(x)

                if (len(x_batch) == self.args.batch_size) | (i == (len(MVTecDataset.image_test[type_test]) - 1)):
                    with torch.no_grad():
                        _ = self(torch.vstack(x_batch))
                    x_batch = []

            self.f1_test[type_test] = np.vstack(self.features[0::4])
            self.f2_test[type_test] = np.vstack(self.features[1::4])
            self.f3_test[type_test] = np.vstack(self.features[2::4])
            self.fl_test[type_test] = np.vstack(self.features[3::4]).squeeze(-1).squeeze(-1)

    def fit(self):
        self.create_test_features()

        # exec knn by final layer feature vector
        d = self.fl_train.shape[1]
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d, faiss.GpuIndexFlatConfig())
        index.add(self.fl_train)


