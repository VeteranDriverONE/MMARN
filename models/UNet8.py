from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

# if __name__ == "__main__":
#     import losses as losses
# else:
#     import models.losses as losses

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, 
        conv_op=nn.Conv3d,conv_args=None,
        norm_op=nn.BatchNorm3d, norm_op_args=None,
        drop_op=nn.Dropout3d, drop_op_args=None,
        non_line_op=nn.LeakyReLU, non_line_op_args=None,
        is_conv_pool=False):
        super(DoubleConv, self).__init__()
        self.drop_op = drop_op
        if conv_args is None:
            conv_args = {'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}
        if norm_op_args is None:
            norm_op_args = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if drop_op_args is None:
            drop_op_args = {'p': 0.5, 'inplace': True}
        if non_line_op_args is None:
            non_line_op_args = {'negative_slope': 1e-2, 'inplace': True}
        if is_conv_pool:
            conv_args2 = {'kernel_size':3,'stride':2,'padding':1,'dilation': 1, 'bias': True}
        else:
            conv_args2 = conv_args

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)

        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = non_line_op(**non_line_op_args)
        
        if drop_op is not None:
            self.drop1 = drop_op(**drop_op_args)
            self.drop2 = drop_op(**drop_op_args)

    def forward(self,x):
        if self.drop_op is not None:
            x1 = self.non_line1(self.norm1(self.drop1(self.conv1(x))))
            x2 = self.non_line2(self.norm2(self.drop2(self.conv2(x1))))
        else:
            x1 = self.non_line1(self.norm1(self.conv1(x)))
            x2 = self.non_line2(self.norm2(self.conv2(x1)))
        return x2


class Trans(nn.Module):
    def __init__(self,inshape, stage_num=5, freeze=True, deep_supervision=False,
        conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Trans,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        self.stage_num = stage_num
        self.deep_supvision = deep_supervision
        self.conv_pool = conv_pool
        self.up_sample = up_sample
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'size':2, 'scale_factor':2, 'mode':'nearest', 'align_corners':None}
        
        self.encode_blocks = [
            DoubleConv(1,32,32,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(32,64,64,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(64,128,128,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(128,256,256,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(256,512,512,drop_op=None,is_conv_pool=conv_pool),
        ]

        if self.conv_pool is False:
            self.down_pool = [
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
            ]

        self.decode_blocks = [
            DoubleConv(512*2,512,256,drop_op=None),
            DoubleConv(256*2,512,128,drop_op=None),
            DoubleConv(128*2,256,64,drop_op=None),
            DoubleConv(64*2,128,32,drop_op=None),
            DoubleConv(32*2,32,16,drop_op=None)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(256*2,256*2,2,2),
                nn.ConvTranspose3d(128*2,128*2,2,2),
                nn.ConvTranspose3d(64*2,64*2,2,2),
                nn.ConvTranspose3d(32*2,32*2,2,2),
            ]

        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.down_pool = nn.ModuleList(self.down_pool)
        self.up_sample = nn.ModuleList(self.up_sample)

        self.spatialtransformer = SpatialTransformer(inshape)
        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow = conv_fn(16, 3, kernel_size=3, padding=1)
        self.trans = nn.MaxPool3d(2,2,0)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

        self.final_conv = conv_fn(16, 1, kernel_size=3, stride=1, padding=1)
        

    def forward(self, moving, fixed, moving_filename, fixed_filename):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = moving
        y = fixed
        x_filename = moving_filename
        y_filename = fixed_filename
        bs = x.size(0)
        prototype_loss = torch.tensor(0)
        consistency_loss = torch.tensor(0)
        style_loss = torch.tensor(0)

        skip_moving = []
        skip_fixed = []

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            if self.conv_pool is False and d!=self.stage_num-1:
                x = self.down_pool[d](x)
                y = self.down_pool[d](y)
            if d!=self.stage_num-1:
                skip_moving.append(x)
                skip_fixed.append(y)
            for i  in range(bs):
                self.setPrototype(x[i].detach(), str(x_filename[i]), d)
                self.setPrototype(y[i].detach(), str(y_filename[i]), d)
                prototype_loss = prototype_loss + self.Center_loss(y[i], str(y_filename[i]), self.prototypes[d]) \
                        + self.Center_loss(x[i], str(x_filename[i]), self.prototypes[d])

        # X -> Y        
        x = torch.concat([x,self.getPrototypes(y_filename,-1)],dim=1)
        x = self.decode_blocks[0](x)
        
        for u in range(1,self.stage_num):
            x = torch.concat([x,self.getPrototypes(y_filename,-(u+1))], dim=1)
            # x = torch.concat([x,self.getPrototypes(y_filename,-(u+1)),skip_moving[-u]], dim=1)
            x = self.decode_blocks[u](self.up_sample[u-1](x))

        # Y -> X
        y = torch.concat([y,self.getPrototypes(x_filename,-1)],dim=1)
        y = self.decode_blocks[0](y)
        
        for u in range(1,self.stage_num):
            y = torch.concat([y,self.getPrototypes(x_filename,-(u+1))], dim=1)
            # y = torch.concat([y,self.getPrototypes(x_filename,-(u+1)),skip_fixed[-u]], dim=1)
            y = self.decode_blocks[u](self.up_sample[u-1](y))

        x = self.final_conv(x)
        y = self.final_conv(y)

        return x, y

    def setPrototype(self,x,x_filename,stage):
        old_shots = self.pro_dict[stage].get(x_filename,0)
        if old_shots == 0:
            self.prototypes[stage][x_filename] = x
        
        p = self.prototypes[stage][x_filename]
        new_shots = old_shots + 1
        # prototype = (x + p*old_shots) / new_shots
        prototype = (1-self.moment) * x + self.moment * p
        self.pro_dict[stage][x_filename] = new_shots
        self.prototypes[stage][x_filename] = prototype
    
    def getPrototypes(self,x_filename,stage):
        prototypes = []
        for filename in x_filename:
            prototypes.append(self.prototypes[stage][str(filename)])
        return torch.stack(prototypes)

    def cosdist(self,fts,prototype,scaler=20):
        dist = F.cosine_similarity(fts, prototype, dim=1) * scaler
        return dist.detach().cpu().numpy()

    def gram(self, X):
        c, n = X.shape[1], X.numel()//X.shape[1]
        X = X.reshape((c, n))
        return torch.matmul(X, X.T) / (c * n)

    def style_loss(self, moved_prototype, prototype, y_type):
        center_loss = torch.tensor(0)
        for i in range(moved_prototype.size(0)):
            center_loss = center_loss + \
                torch.square(self.gram(moved_prototype[i]) - self.gram(prototype[str(y_type[i])])).mean()
        return center_loss

    def mse_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def Center_loss(self, source_ft,source_file,source_pros):
        source_pro = source_pros[source_file]
        # center_loss = torch.mean(torch.square((source_ft-source_pro)))
        center_loss = torch.pow((source_ft-source_pro),2).mean()
        return center_loss

    def cal_attention(self, x, fixed, prototype):
        b,c,z,h,w = x.size()
        x = x.view(b,c,-1).transpose(-2,-1) # b,zhw,c
        fixed = fixed.view(b,c,-1) # b,c,zhw
        prototype = prototype.view(b,c,-1)
        x_f = torch.matmul(x,fixed).unsqueeze(1) # b,1,zhw,zhw
        x_p = torch.matmul(x,prototype).unsqueeze(1) 
        soft_x_pf = F.softmax(torch.concat([x_f,x_p],1),dim=1) # b,2,zhw,zhw
        w_f = torch.matmul(soft_x_pf[:,0,:,:],fixed.transpose(-2,-1))
        w_pro = torch.matmul(soft_x_pf[:,1,:,:],prototype.transpose(-2,-1))
        return (w_f + w_pro).transpose(-2,-1).view(b,c,z,h,w)


class Reg(nn.Module):
    def __init__(self,inshape, stage_num=5, freeze=True, deep_supervision=False,
        conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Reg,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        self.stage_num = stage_num
        self.deep_supvision = deep_supervision
        self.conv_pool = conv_pool
        self.up_sample = up_sample
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'size':2, 'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        self.encode_blocks = [
            DoubleConv(1,32,32,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(32,64,64,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(64,128,128,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(128,256,256,drop_op=None,is_conv_pool=conv_pool),
            DoubleConv(256,512,512,drop_op=None,is_conv_pool=conv_pool),
        ]
        if self.conv_pool is False:
            self.down_pool = [
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
                nn.MaxPool3d(**pool_args),
            ]

        self.decode_blocks = [
            DoubleConv(512+256,512,128,drop_op=None),
            DoubleConv(128*2,256,64,drop_op=None),
            DoubleConv(64*2,128,32,drop_op=None),
            DoubleConv(32*2,32,16,drop_op=None)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(512+256,512+256,2,2),
                nn.ConvTranspose3d(256,256,2,2),
                nn.ConvTranspose3d(128,128,2,2),
                nn.ConvTranspose3d(64,64,2,2),
            ]

        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.down_pool = nn.ModuleList(self.down_pool)
        self.up_sample = nn.ModuleList(self.up_sample)

        self.spatialtransformer = SpatialTransformer(inshape)
        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow = conv_fn(16, 3, kernel_size=3, padding=1)
        self.trans = nn.MaxPool3d(2,2,0)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

        self.final_conv = conv_fn(16, 1, kernel_size=3, stride=2, padding=1) 

    def forward(self, moving, fixed):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = moving
        y = fixed

        skip_moving = []
        skip_fixed = []
        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            if self.conv_pool is False and d!=self.stage_num-1:
                x = self.down_pool[d](x)
                y = self.down_pool[d](y)
            if d!=self.stage_num-1:
                skip_moving.append(x)
                skip_fixed.append(y)

        # X -> Y
        for u in range(self.stage_num-1):
            x = torch.concat([x, skip_fixed[-(u+1)]], dim=1)
            # x = torch.concat([x,self.getPrototypes(y_filename,-(u+1)),skip_moving[-u]], dim=1)
            x = self.decode_blocks[u](self.up_sample[u](x))

        # Y -> X
        for u in range(self.stage_num-1):
            y = torch.concat([y, skip_moving[-(u+1)]], dim=1)
            # y = torch.concat([y,self.getPrototypes(x_filename,-(u+1)),skip_fixed[-u]], dim=1)
            y = self.decode_blocks[u](self.up_sample[u](y))

        x = self.flow(x)
        y = self.flow(y)

        return x, y


class PatchEmbed(nn.Module):
    # LinearEmbeding 转换图像为patch，没噶patch拉伸成向量，以通道的形式表示，并将patch的元素转换为D维
    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=96, norm_layer=None):
        super(PatchEmbed,self).__init__()
        
        self.img_size = (img_size, img_size, img_size)  # 3D: (img_siz, img_size, img_size)
        self.patch_size = (patch_size, patch_size, patch_size)
        self.patches_resolution = [img_size[0]//patch_size[0], img_size[1]//patch_size[1], img_size//patch_size[2]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # 转换成D维，patch不重叠
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm - None 

    def forward(self, x):
        x = self.proj(x)  # B, embed_dim, num_patch0, num_patch1, num_patch2
        x = torch.flatten(x, 2)  # B,C,ZHW
        x = torch.permute(x, 0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)

        return x

class PatchMerging(nn.Module):
    # 将分辨率转换为通道；将长宽高缩小二分之一，并将其转为通道，因此通道是原来的8倍
    def __init__(self, input_res, dim, norm_layer=nn.layerNorm):
        super(PatchMerging,self).__init__()
        self.input_res = input_res
        self.dim = dim
        self.norm_layer = norm_layer
        self.reduction = nn.Liner(8*dim, 4*dim, bias = False)

    def forward(self, x):
        z, h, w = self.input_res
        b, l, c = x.shape
        assert l==h*w, '分辨率错误'
        assert z%2==0 and h%2 == 0 and w%2==0, '分辨率必须为2的整数倍'
        x = x.view(b, z, h, w, c) # 还原分辨率
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 0::2, 0::2, 1::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7,x7],-1) # B,z/2,h/2,w/2,4,c
        x = x.view(b, -1, 8*c) # 将分辨率重构成原来的二分之一，通道是原来的8倍

# 划分窗口b,z,h,w,c -> b*w_num, 
def window_partition(x, window_size):
    b, z, h, w, c = x.shape
    x = x.view(b,z//window_size, window_size, h//window_size, window_size, w//window_size, window_size, c)
    windows = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, window_size, window_size, window_size, c)
    return windows

# 将b*num, z,h,w,c原回b*w_num
def window_reverse(windows, window_size, z, h, w):
    # windows shape: num_win*B, window_size, window_size, window_size, C
    b = int(windows.shape[0] / (z*h*w / window_size**3))
    x = windows.view(b, z//window_size, h//window_size, w//window_size, window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,5,2,4,6,7).contiguous().view(b,z,h,w,-1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()

        self.dim = dim 
        self.window_size = window_size # Z, H, W
        self.num_heads = num_heads # 注意力头
        head_dim = dim // num_heads # 每个注意力头的通道数 = 总通道数 / 注意力头
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # 设置一个形状为（2*(Wz-1) * 2*(Ww-1) * 2*(Ww-1), nH）的可学习变量，用于后续的位置编码
        self.relative_position_bias_table = nn.parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1)* (2 * window_size[2] - 1), num_heads))
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wz-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_z = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.meshgrid([coords_z, coords_h, coords_w]) # -> 3*(wz, wh, ww)

        coords = torch.stack(coords)  # 3, Wz, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wz*Wh*Ww

        relative_coords_first = coords_flatten[:, :, None]  # 2, wh*ww, 1
        relative_coords_second = coords_flatten[:, None, :] # 2, 1, wh*ww
        relative_coords_third = corrds_flatten[]
        relative_coords = relative_coords_first - relative_coords_second # 最终得到 2, wh*ww, wh*ww 形状的张量

        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1


        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


if __name__ == "__main__":
    trans = Trans()
    reg = Reg()
    moving = torch.rand(1,1,64,128,192)
    fixed = torch.rand(1,1,64,128,192)
    mf_trans, fm_trans = trans(moving, fixed ['CMP'], ['UP'])
    mf_reg, fm_reg = reg(moving,fixed)
    
    print('运行结束')
