import torch
import torch.nn as nn

# basic building block for ResNet-18, ResNet-34
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides, 
                 conv_op=nn.Conv3d, conv_args={},
                 norm_op=nn.BatchNorm3d, norm_args={}, 
                 non_line_op=nn.ReLU, non_line_args={'negative_slope': 1e-2, 'inplace': True},
                 is_se=False):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        self.conv1 = conv_op(in_channels, out_channels, 3, stride=strides, padding=1, bias=False)  # same padding
        self.conv2 = conv_op(out_channels, out_channels, 3, stride=1, padding=1, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels, 16)

        # fit input with residual output
        self.short_cut = nn.Sequential()
        if strides is not 1:
            self.short_cut = nn.Sequential(
                conv_op(in_channels, out_channels, 1, stride=strides, padding=0, bias=False),
                norm_op(out_channels, **norm_args)
            )
        self.non_line = non_line_op(**non_line_args)
        self.last_ch = out_channels

    def forward(self, x):
        out = self.non_line(self.conv1(x))
        out = self.conv2(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.short_cut(x)
        return self.non_line(out)


# BottleNeck block for RestNet-50, ResNet-101, ResNet-152
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, strides, 
                conv_op = nn.Conv3d, conv_args = {},
                norm_op = nn.BatchNorm3d, norm_args = {},
                non_line_op=nn.ReLU, non_line_args={'negative_slope': 1e-2, 'inplace': True},
                is_se=False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = conv_op(in_channels, out_channels, 1, stride=1, padding=0, bias=False)  # same padding
        self.conv2 = conv_op(out_channels, out_channels, 3, stride=strides, padding=1, bias=False)
        self.conv3 = conv_op(out_channels, 4*out_channels, 1, stride=1, padding=0, bias=False, activation=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        # fit input with residual output
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, 1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.last_ch = out_channels * 4
        self.non_line = non_line_op(**non_line_args)

    def forward(self, x):
        out = self.non_line(self.conv1(x))
        out = self.non_line(self.conv2(out))
        out = self.conv3(out)
        if self.is_se:
            coefficient = self.se(out)
            out = out * coefficient
        out = out + self.shortcut(x)
        return self.non_line(out)


class ResNet(nn.modules):
    def __init__(self, in_ch, out_ch, block, groups):
        super(ResNet, self).__init__()
        self.block = block
        base_ch = 64
        self.cur_ch = base_ch
        self.layer1 = nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=7, stride=2),
                                    nn.BatchNorm3d(base_ch),
                                    nn.ReLU(),
                                    nn.MaxPool3d(3,2,1))
        self.layer2 = self.__make_layer__(base_ch, groups[0], 1)
        self.layer3 = self.__make_layer__(2*base_ch, groups[1], 2)
        self.layer4 = self.__make_layer__(4*base_ch, groups[2], 2)
        self.layer5 = self.__make_layer__(8*base_ch, groups[3], 2)

        self.final_conv = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)),
                                        nn.Linear(self.cur_ch, out_ch),
                                        nn.Softmax(dim=1))

    def _make_conv_x(self, channels, block_num, strides, index):
        """
        making convolutional group
        :param channels: output channels of the conv-group
        :param blocks: number of blocks in the conv-group
        :param strides: strides
        :return: conv-group
        """
        blocks = [self.block(self.cur_ch, channels, strides)]
        for i in range(1,block_num):
            block = self.block(channels, channels, 1)
            blocks.append(block)
            self.ch = block.last_ch

        blocks = nn.ModuleList(blocks)
        return blocks

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.final_conv(x)
        return x


def def_resnet18(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)

def def_resnet34(num_classes=1000):
    return ResNet(block=BasicBlock, groups=[3, 4, 6, 3], num_classes=num_classes)

def def_resnet50(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)

def def_resnet101(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 4, 23, 3], num_classes=num_classes)

def def_resnet152(num_classes=1000):
    return ResNet(block=BottleNeck, groups=[3, 8, 36, 3], num_classes=num_classes)


if __name__ == '__main__':
    resnet18 = def_resnet18()
    resnet34 = def_resnet34()
    resnet50 = def_resnet50()
    resnet101 = def_resnet101()
    resnet152 = def_resnet152()
    
    input = torch.rand((2,1,224,224,224))
    output = resnet18(input)
    print(output.shape)


