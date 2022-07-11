import torch
import torchvision
from torch.nn import functional as F
import numpy as np


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        X = (X + 1) / 2
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGG19LOSS(torch.nn.Module):
    def __init__(self):
        super(VGG19LOSS, self).__init__()
        self.model = VGG19()

    def forward(self, fake, target, weight_mask=None, loss_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        vgg_fake = self.model(fake)
        vgg_target = self.model(target)

        value_total = 0
        for i, weight in enumerate(loss_weights):
            value = torch.abs(vgg_fake[i] - vgg_target[i].detach())
            if weight_mask is not None:
                bs, c, H1, W1 = value.shape
                _, _, H2, W2 = weight_mask.shape
                if H1 != H2 or W1 != W2:
                    cur_weight_mask = F.interpolate(weight_mask, size=(H1, W1))
                    value = value * cur_weight_mask
                else:
                    value = value * weight_mask
            value = torch.mean(value, dim=[x for x in range(1, len(value.size()))])
            value_total += loss_weights[i] * value
        return value_total
