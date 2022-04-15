import torch.nn as nn
import torch
import torch.nn.functional as F

from models import loss


def initialize_weights(layer, activation='relu'):

    for module in layer.modules():
        module_name = module.__class__.__name__

        if activation in ('relu', 'leaky_relu'):
            layer_init_func = nn.init.kaiming_uniform_
        elif activation == 'tanh':
            layer_init_func = nn.init.xavier_uniform_
        else:
            raise Exception('Please specify your activation function name')

        if hasattr(module, 'weight'):
            if module_name.find('Conv2') != -1:
                layer_init_func(module.weight)
            elif module_name.find('BatchNorm') != -1:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0.0)
            elif module_name.find('Linear') != -1:
                layer_init_func(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.1)
            else:
                # print('Cannot initialize the layer :', module_name)
                pass
        else:
            pass


class PoreNet_SC(nn.Module):
    def __init__(self):
        super(PoreNet_SC, self).__init__()

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        ])

        self.decoder = nn.Sequential(*[
            nn.Conv2d(40, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ])

        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.decoder)

    def forward(self, x):
        _, _, h, w = x.shape

        feat_s1 = self.conv1(x)
        feat_s2 = self.conv2(feat_s1)
        feat_s3 = self.conv3(feat_s2)

        feat_cat = torch.cat((feat_s1, feat_s2, feat_s3), dim=1)
        output = self.decoder(feat_cat)

        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

        return output


class PoreNet_fullySC(nn.Module):
    def __init__(self):
        super(PoreNet_fullySC, self).__init__()

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(8, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        ])

        self.decoder = nn.Sequential(*[
            nn.Conv2d(72, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        ])

        initialize_weights(self.conv1)
        initialize_weights(self.conv2)
        initialize_weights(self.conv3)
        initialize_weights(self.decoder)

    def forward(self, x):
        _, _, h, w = x.shape

        feat_s1 = self.conv1(x)

        feat_s2 = self.conv2(feat_s1)
        feat_s2 = torch.cat((feat_s1, feat_s2), dim=1)

        feat_s3 = self.conv3(feat_s2)
        feat_cat = torch.cat((feat_s1, feat_s2, feat_s3), dim=1)
        output = self.decoder(feat_cat)

        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

        return output
