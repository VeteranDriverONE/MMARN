import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

class Discriminator(nn.Module):
    def __init__(self, **args):
        super(Discriminator,self).__init__()
        for k,v in args.items():
            setattr(self, k, v)
        
        self.model = resnet34()
        last_input_ch = 256 # in_features=2048
        self.fc = nn.Sequential(
            nn.Linear(last_input_ch, 64),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Dropout(0.2),
        )
        self.aap = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, x):
        _, _, _, c4 = self.model(x)
        out = self.fc(self.aap(c4).flatten(1,-1))
        out = out.mean(0)
        return out.view(1)


class ResNet(nn.Module):
    def __init__(self, block, layers, in_ch=1, shortcut_type='B'):
        super(ResNet, self).__init__()
        self.in_ch = 64 # 初始通道
        
        # self.conv1 = nn.Conv3d(in_ch,64,kernel_size=7,stride=(1, 2, 2),padding=(3, 3, 3),bias=False)
        self.conv1 = nn.Conv3d(in_ch,64,kernel_size=7,stride=2,padding=(3, 3, 3),bias=False) # 1->64,32,64,64
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type) # Res50:64->256,32,64,64; Res34:64->64,32,64,64
        self.layer2 = self._make_layer(block, 64, layers[1], shortcut_type, stride=2) # Res50:256->512,16,32,32; Res34:64,16,32,32
        self.layer3 = self._make_layer(block, 128, layers[2], shortcut_type, stride=2) # 512->1024,8,16,16
        self.layer4 = self._make_layer(block, 256, layers[3], shortcut_type, stride=2) # 1024->2048,4,8,8

        # self.layer1_cat = nn.Conv3d(256+64, 256, kernel_size=1, stride=1)
        # self.layer2_cat = nn.Conv3d(256+128, 256, kernel_size=1, stride=1)
        # self.layer3_cat = nn.Conv3d(512+256, 512, kernel_size=1, stride=1)
        # self.layer4_cat = nn.Conv3d(1024+512, 1024, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_ch, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch * block.expansion:
            # stride！=1表示采用卷积层直接下采样，因此需要调整残差路线的大小，
            # 输入通道不是输出通道的四倍时也需要调整残差劣解的通道数
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, out_ch=out_ch * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.in_ch, out_ch * block.expansion, kernel_size=1, stride=stride, bias=False), 
                    nn.BatchNorm3d(out_ch * block.expansion))
        layers = []
        layers.append(block(self.in_ch, out_ch, stride, downsample))  # 
        self.in_ch = out_ch * block.expansion  # in_ch是out_ch的四倍，否则残差通道不匹配；且最后一个block的输出通道，即为下一层的输入
        for i in range(1, blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        c1 = self.layer1(x)
        # c1 = self.layer1_cat(torch.concat([xs[3],c1],dim=1))
        
        c2 = self.layer2(c1)
        # c2 = self.layer2_cat(torch.concat([xs[2],c2],dim=1))
        
        c3 = self.layer3(c2)
        # c3 = self.layer3_cat(torch.concat([xs[1],c3],dim=1))
        
        c4 = self.layer4(c3)
        # c4 = self.layer4_cat(torch.concat([xs[0],c4],dim=1))
        return c1, c2, c3, c4


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 两次卷积，一次下采样
        self.stride = stride
        self.conv1 = conv3x3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        
        self.conv3 = nn.Conv3d(out_ch, out_ch * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample # shutcut，残差通道切换
        self.stride = stride

    def forward(self, x):
        residual = x

        # 用1*1卷积压缩通道
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 用1成1卷积得到新的输出通道
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3x3(in_ch, out_ch, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_ch,out_ch,kernel_size=3,
                    stride=stride,padding=1,bias=False)


def downsample_basic_block(x, out_ch, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), out_ch - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out

def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

if __name__ == '__main__':
    dis = Discriminator()
