import torch
from torchinfo import summary
from torch import nn


class Spade(nn.Module):
    features = []

    def __init__(self, args):
        super(Spade, self).__init__()
        self.args = args

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

    def hook(self, input, output):
        Spade.features.append(output.detach().cpu().numpy())

    def forward(self, img):
        Spade.features.clear()
        self.backbone(img)

        return Spade.features
