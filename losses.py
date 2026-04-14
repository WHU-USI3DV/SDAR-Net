import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        L1_temp = 0.8 * self.L1(xs, ys)
        L_total = L1_temp + L2_temp
        return L_total

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)
