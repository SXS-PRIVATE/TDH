# -*- coding: utf-8 -*-
# @Time    : 2019/7/21
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
import torch.nn as nn
import torchvision.models as models
from torchcmh.models import BasicModule

__all__ = ['AlexNet7', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

alexnet = models.alexnet(pretrained=True)


class AlexNet7(BasicModule):
    def __init__(self, num_classes=1000):
        super(AlexNet7, self).__init__()
        self.module_name = "Alexnet"
        self.pre_layer = nn.Sequential(
             *list(alexnet.features.children())[:-1]  # 获取前7层
        )
        self.code_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        features_7 = self.pre_layer(x)
        features = self.code_layer(features_7)
        return features_7, features


def alexnet(pretrained=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet7(**kwargs)
    if pretrained:
        model.init_pretrained_weights(model_urls['alexnet'])
    return model
