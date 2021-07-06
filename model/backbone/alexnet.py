import torch
import torch.nn as nn
from collections import OrderedDict
from model.backbone.resnet import Backbone
import os


class AlexNet(Backbone):

    def __init__(self, output_layers, size=1, frozen_layers=(), activate_layers=()):
        super(AlexNet, self).__init__(frozen_layers=frozen_layers, activate_layers=activate_layers)
        self.output_layers = output_layers
        configs = [3, 64, 192, 384, 256, 256]
        configs = list(map(lambda x: 3 if x == 3 else x * size, configs))
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(configs[1], configs[2], kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(configs[2], configs[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()
        if output_layers is None:
            output_layers = self.output_layers
        x = self.layer1(x)
        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)
        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        raise ValueError('output_layer is wrong.')

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


def alexnet(output_layers=None, pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['layer1', 'layer2', 'layer3']:
                raise ValueError('Unknown layer: {}'.format(l))
    model = AlexNet(output_layers, **kwargs)

    if pretrained:
        current_file = os.path.dirname(os.path.realpath(__file__))
        state_dict = torch.load(os.path.join(current_file, '../pretrained_models/alexnet.pth'))
        model.load_state_dict(state_dict)
    return model
