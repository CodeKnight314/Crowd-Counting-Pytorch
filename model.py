import torch 
import torch.nn as nn 
from torchvision.models import vgg19

class CSRNet(nn.Module): 
    def __init__(self, config : str = "B"):
        super().__init__()
        
        self.frontend = vgg19(weights = 'IMAGENET1K_V1').features[0:23]
        for params in self.frontend.parameters(): 
            params.requires_grad = False

        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        
        if config == "A":
            self.d_rate = [1, 1, 1, 1, 1, 1]
        elif config == "B": 
            self.d_rate = [2, 2, 2, 2, 2, 2]
        elif config == "C": 
            self.d_rate = [2, 2, 2, 4, 4, 4]
        elif config == "D": 
            self.d_rate = [4, 4, 4, 4, 4, 4]

        self.backend = self.make_layers()
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x): 
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, in_channels = 512, batch_norm=True):
        layers = []
        for v, d_rate in zip(self.backend_feat, self.d_rate):
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        return nn.Sequential(*layers)