import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch as th
import torchvision.models as models

# MODEL FROM CILRS
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_shape, num_classes=1000):

        im_channels, im_h, im_w = input_shape

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(im_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=0)

        # TODO: THis is a super hardcoding ..., in order to fit my image size on resnet
        # if block.__name__ == 'Bottleneck':
        #     self.fc = nn.Linear(6144, num_classes)  # 6144 original,
        # else:
        #     self.fc = nn.Linear(1536, num_classes)  # 1536 original, 8192, 9216

        with th.no_grad():
            x = th.zeros(1, im_channels, im_h, im_w)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x0 = self.maxpool(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

            x = self.avgpool(x4)
            x = x.view(x.size(0), -1)
            n_flatten = x.shape[1]
        # print(n_flatten)
        self.fc = nn.Linear(n_flatten, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        # print ('in resnet: ', x.shape)
        x = self.fc(x)

        return x

    def get_layers_features(self, x):
        # Just get the intermediate layers directly.

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.avgpool(x4)
        x = x5.view(x.size(0), -1)
        x = self.fc(x)

        all_layers = [x0, x1, x2, x3, x4, x5, x]
        return all_layers


def resnet34_cilrs(input_shape, num_classes, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], input_shape, num_classes)
    if pretrained:
        model_dict = model_zoo.load_url(model_urls['resnet34'])
        # remove the fc layers
        del model_dict['fc.weight']
        del model_dict['fc.bias']
        state = model.state_dict()
        state.update(model_dict)
        model.load_state_dict(state)
    return model

# MODEL FROM TORCH MODEL ZOO


def get_model(model_name, input_shape, num_classes, pretrained=False):
    print(f'Loading resnet, model_name={model_name}, pretrained={pretrained}')

    im_channels, im_t, im_h, im_w = input_shape
    if model_name == 'resnet34_cilrs':
        if im_channels == 3:
            model = resnet34_cilrs(input_shape, num_classes, pretrained=pretrained)
        else:
            model = resnet34_cilrs(input_shape, num_classes, pretrained=False)
    else:
        assert model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
                              'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
                              'video.mc3_18', 'video.r2plus1d_18', 'video.r3d_18']
        if 'video' in model_name:
            ResnetModule = getattr(models.video, model_name.split('.')[1])
        else:
            ResnetModule = getattr(models, model_name)
            im_channels = im_channels*im_t
        model = ResnetModule(pretrained=pretrained, progress=True)

        if im_channels != 3:
            assert 'video' not in model_name
            print(f'Mismatch {model_name} first conv input channel. desired:{im_channels}, predefined:3')
            old = model.conv1
            model.conv1 = th.nn.Conv2d(
                im_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias)

        if num_classes != model.fc.out_features:
            print(f'Mismatch {model_name} last fc output dim. desired:{num_classes}, predefined:{model.fc.out_features}')
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
    return model


# def resnet50(input_shape, num_classes, pretrained=False):
#     im_channels, im_h, im_w = input_shape

#     if im_channels == 3:
#         model = models.resnet50(pretrained=pretrained, progress=True)
#     else:
#         model = models.resnet50(pretrained=False)
#         old = model.conv1
#         model.conv1 = th.nn.Conv2d(
#             im_channels, old.out_channels,
#             kernel_size=old.kernel_size, stride=old.stride,
#             padding=old.padding, bias=old.bias)
#     return model
