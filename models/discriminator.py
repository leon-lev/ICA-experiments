import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_channels_in, n_classes):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(n_channels_in, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, n_classes, 4, padding=1)]
        model += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        x =  self.model(x)
        logits = x.squeeze()
        return logits 