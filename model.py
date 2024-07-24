import torch 
import torch.nn as nn 
from torchvision.models import vgg19

class CRSNet(nn.Module): 
    def __init__(self):
        super().__init__()
        
        # load pretrained vgg19 model weights
        self.frontend = vgg19(weights = 'IMAGENET1K_V1').features[0:23]
        for params in self.frontend.parameters(): 
            params.requires_grad = False

        self.backend_feat  = [512, 512, 512, 256, 128, 64]
        self.backend = self.make_layers()
        
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x): 
        x = self.frontend(x)
        print(x.shape )
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def make_layers(self, in_channels = 512, batch_norm=False, dilation = True):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in self.backend_feat:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)