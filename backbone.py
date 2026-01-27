import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet34(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3, out_channels_for_pa=32): # <-- 添加参数
    print(in_c)
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    if in_c != 3:
        pretrained = False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model

def ResNet18(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3, out_channels_for_pa=32): # <-- 添加参数
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    if in_c !=3:
        pretrained=False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet18'])
    return model

def ResNet50(output_stride, BatchNorm=nn.BatchNorm2d, pretrained=True, in_c=3, out_channels_for_pa=32): # <-- 添加参数
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    if in_c !=3:
        pretrained=False
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model


class BasicBlock(nn.Module):
    expansion = 1
    """ ResNet-18 和 ResNet-34 使用的基本残差块"""
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.do1 = nn.Dropout2d(p=0.2)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation   #空洞卷积
        # ============================================
        # 新增: 实例化 SELayer
        # SELayer 的 channel 参数应该是 Bottleneck 的输出通道数
        self.se_layer = SELayer(planes * 4)
        # ============================================

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # ============================================
        # 新增: 在残差连接和最终 ReLU 之前应用 SELayer
        out = self.se_layer(out)
        # ============================================

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PA(nn.Module):
    def __init__(self, inchan = 512, out_chan = 32):
        super().__init__()
        self.conv = nn.Conv2d(inchan, out_chan, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.re = nn.ReLU()
        self.do = nn.Dropout2d(0.2)

        self.pa_conv = nn.Conv2d(out_chan, out_chan, kernel_size=1, padding=0, groups=out_chan)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv(x)
        x = self.do(self.re(self.bn(x0)))
        return x0 *self.sigmoid(self.pa_conv(x))  #将 x (经过处理的特征) 通过一个逐通道的 1x1 卷积 (pa_conv)，然后经过 Sigmoid 激活函数生成注意力图。最后，将这个注意力图元素级地乘以原始转换后的特征 x0，实现特征的加权。


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, in_c=3, out_channels_for_pa=32):

        self.inplanes = 64
        self.in_c = in_c
        # print('in_c: ',self.in_c)
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 4:
            strides = [1, 1, 1, 1]
            dilations = [1, 2, 4, 8]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.in_c, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()



        # 根据传入的block类型动态确定PA模块的输入通道数
        if block == BasicBlock:  # 对应 ResNet-18 或 ResNet-34
            pa_s16_in_channels = 512  # layer4 的输出通道 (对于 BasicBlock 通常是 512)
            pa_s8_in_channels = 128  # layer2 的输出通道
            pa_s4_in_channels = 64  # layer1 的输出通道
        elif block == Bottleneck:  # 对应 ResNet-50 或 ResNet-101
            # 注意：这里需要根据 output_stride 来判断 layer3 还是 layer4 的输出作为 s16
            # 在你的 ResNet 类中，out_s16 总是来自 layer4 的最终输出
            # 所以我们用 layer4 的最终通道数
            pa_s16_in_channels = 512 * Bottleneck.expansion  # layer4 的输出通道 (512 * 4 = 2048)
            pa_s8_in_channels = 128 * Bottleneck.expansion  # layer2 的输出通道 (128 * 4 = 512)
            pa_s4_in_channels = 64 * Bottleneck.expansion  # layer1 的输出通道 (64 * 4 = 256)
        else:
            raise ValueError(
                "Unsupported block type for PA initialization. Only BasicBlock and Bottleneck are supported.")

        # 使用动态确定的通道数来初始化 PA 模块
        self.pos_s16 = PA(pa_s16_in_channels, out_channels_for_pa)
        self.pos_s8 = PA(pa_s8_in_channels, out_channels_for_pa)
        self.pos_s4 = PA(pa_s4_in_channels, out_channels_for_pa)

    """创建 ResNet 的标准阶段"""
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    """通过在同一阶段使用不同空洞率的卷积，可以捕获不同尺度的上下文信息，而不会引入额外的下采样，这对于密集预测任务非常有利。"""
    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    "数据流经骨干网络的路径"
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # | 4


        x = self.layer1(x)  # | 4
        low_level_feat2 = x

        x = self.layer2(x)  # | 8
        low_level_feat3 = x

        x = self.layer3(x)  # | 16
        low_level_feat4 = x  # 新增：保存 layer3 的输出

        x = self.layer4(x)  # | 16？  32

        out_s16, out_s8, out_s4 = self.pos_s16(x), self.pos_s8(low_level_feat3), self.pos_s4(low_level_feat2)
        return out_s16, out_s8, out_s4


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def build_backbone(backbone, output_stride, BatchNorm, in_c=3, out_channels_for_pa=32): # <-- 添加参数
    if backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    elif backbone == 'resnet34':
        return ResNet34(output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    elif backbone == 'resnet18':
        return ResNet18(output_stride, BatchNorm, in_c=in_c, out_channels_for_pa=out_channels_for_pa) # <-- 传递参数
    else:
        raise NotImplementedError

