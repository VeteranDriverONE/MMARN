from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import unfoldNd
from models.losses import ContrastiveLoss
from torch.distributions.normal import Normal
from models.InfoNCE import InfoNCE, InfoNCELoss, InfoNCE2, normalize

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, 
        conv_op=nn.Conv3d,conv_args={'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}, conv_args2=None,
        norm_op=nn.BatchNorm3d, norm_op_args={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        non_line_op=nn.LeakyReLU, non_line_op_args={'negative_slope': 1e-2, 'inplace': True},
        first_op=None, first_op_args=None,
        last_op=None, last_op_args=None):

        super(DoubleConv, self).__init__()

        self.first_op = first_op
        self.last_op = last_op

        if conv_args2 is None:
            conv_args2 = conv_args

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)

        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = non_line_op(**non_line_op_args)
        
        if first_op is not None:
            self.first_op = first_op(**first_op_args)

        if last_op is not None:
            self.last_op = last_op(**last_op_args)

    def forward(self,x):
        if self.first_op is not None:
            x = self.first_op(x)

        x1 = self.non_line1(self.norm1(self.conv1(x)))
        x2 = self.non_line2(self.norm2(self.conv2(x1)))

        if self.last_op is not None:
            x2 = self.last_op(x2)

        return x2

class Encoder(nn.Module):
    # 采用风格
    def __init__(self,inshape, in_ch=1, out_ch=1, stage_num=5, types=('CMP','NP','UP'),
        conv_pool=False, pool_args=None,  freeze_e1=False, freeze_e2=False):
        super(Encoder, self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = True

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
            encode_blocks2 = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
        else:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]
            encode_blocks2 = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]


        self.encode_blocks = nn.ModuleList(encode_blocks) # 模态
        self.encode_blocks2 = nn.ModuleList(encode_blocks2) # 形态

        # self.contra_fn = ContrastiveLoss(1)
        self.triplet_fn = nn.TripletMarginLoss(5, p=2)

        input = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        prototypes = []
        self.masks = {}
        self.numel = []
        self.k_size = [9, 5, 3, 3, 3]
        for d in range(self.stage_num):
            input = self.encode_blocks[d](input)
            gap = self.k_size[d] // 2
            if d > 2:
                mask = []
                for z in range(input.shape[2]):
                    for h in range(input.shape[3]):
                        for w  in range(input.shape[4]):
                            tmp_mask = torch.zeros_like(input[0,0],dtype=torch.bool)
                            tmp_mask[max(z-gap,0):z+gap+1,max(h-gap,0):h+gap+1,max(w-gap,0):w+gap+1] = True
                            mask.append(tmp_mask.flatten())

                mask = (torch.stack(mask).unsqueeze(0) * ~torch.eye(input[0,0].numel(),dtype=bool)).cuda()
                mask_idx = torch.nonzero(mask).T  # 这里需要转置一下
                mask_data = mask[mask_idx[0], mask_idx[1], mask_idx[2]]
                coo_mask = torch.sparse_coo_tensor(mask_idx, mask_data, mask.shape)

                self.masks[str(d)] = coo_mask
                # self.numel.append(self.masks[str(d)].sum(-1).byte())
            prototype = {}
            for k in types:
                # prototype[k] = nn.Parameter(torch.rand_like(input.squeeze(0)))
                prototype[k] = nn.Parameter(torch.rand((input.shape[1], input.shape[1])))
                # prototype[k] = torch.rand((input.shape[1], input.shape[1]))
            prototypes.append(nn.ParameterDict(prototype))
            # prototypes.append(prototype[k])
        self.prototypes = nn.ParameterList(prototypes)
        # self.register_buffer('prototypes', prototypes)

        for d in range(self.stage_num):
            if freeze_e1:
                for _, p in self.encode_blocks[d].named_parameters():
                    p.requires_grad = False

            if freeze_e2:
                for _, p in self.encode_blocks2[d].named_parameters():
                    p.requires_grad = False
            
            if freeze_prototypes:
                for _, p in self.prototypes[d].items():
                    p.requires_grad = False

    def set_input(self, moving, fixed, moving_filename, fixed_filename):
        self.moving = moving
        self.fixed = fixed
        self.m_type = moving_filename
        self.f_type = fixed_filename

    def forward1(self):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = self.moving
        x2 = self.moving
        y = self.fixed
        y2 = self.fixed
        bs = x.shape[0]
        x_type = self.m_type
        y_type = self.f_type

        prototype_loss = torch.tensor(0)
        contra_loss = torch.tensor(0)

        self.skip_content_moving = []
        self.skip_content_fixed = []
        self.skip_proto_moving =[]
        self.skip_proto_fixed =[]

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            x2 = self.encode_blocks2[d](x2)
            y2 = self.encode_blocks2[d](y2)
            
            self.skip_proto_moving.append(x)
            self.skip_proto_fixed.append(y)
            self.skip_content_moving.append(x2)
            self.skip_content_fixed.append(y2)
            
            x_gram = self.gram2(x)
            y_gram = self.gram2(y)

            if d>1:
                prototype_loss = prototype_loss + (x_gram - self.getPrototypes(x_type, d)).square().mean() \
                            + (y_gram - self.getPrototypes(y_type, d)).square().mean()
                # contra_loss = contra_loss + self.contra_fn(x_gram, y_gram, 1) # 对比损失
                contra_loss  = contra_loss + self.triplet_fn(x_gram, self.getPrototypes(x_type, d), self.getPrototypes(y_type, d)) \
                                            + self.triplet_fn(y_gram, self.getPrototypes(y_type, d), self.getPrototypes(x_type, d))

            for i  in range(bs):
                self.setPrototype(x_gram[i].detach(), str(x_type[i]), d)
                self.setPrototype(y_gram[i].detach(), str(y_type[i]), d)
        
        self.proto_loss = prototype_loss
        self.contra_loss = contra_loss
        return  self.skip_proto_moving, self.skip_proto_fixed, self.skip_content_moving, self.skip_content_fixed

    
    def forward2(self, x_recon):
        # encoder again
        warped_x = x_recon
        warped_x2 = x_recon

        content_loss = torch.tensor(0)
        style_loss = torch.tensor(0)
        for d in range(self.stage_num):
            warped_x = self.encode_blocks[d](warped_x)
            warped_x2 = self.encode_blocks2[d](warped_x2)
            
            # style_loss = style_loss + (self.gram2(warped_x) - self.getPrototypes(y_filename, d)).square().mean()
            # style_loss = style_loss + (self.gram(warped_x) - self.gram(skip_proto_fixed[d])).square().mean()
            if d > 1: 
                style_loss = style_loss + (self.gram2(warped_x) - self.getPrototypes(self.f_type, d)).square().mean()

            if d >= 1:
                geo_mx = self.geo_loss3(warped_x2, d) # m->f, B,C,ZHW,ZHW
                geo_fx = self.geo_loss3(self.skip_content_moving[d], d)
                cos_sim = F.cosine_similarity(geo_mx, geo_fx, dim=-1)
                content_loss = content_loss + torch.exp(1-cos_sim).mean()


            # if d > 2:
            #     geo_mx = self.geo_loss2(warped_x2, d) # m->f, B,C,ZHW,ZHW
            #     geo_fx = self.geo_loss2(self.skip_content_moving[d], d)

            #     tmp_we1 = geo_mx.square().sum(-1).sqrt()
            #     tmp_we2 = geo_fx.square().sum(-1).sqrt()
            #     we1_s = (torch.softmax(tmp_we1/tmp_we1.max(), dim=1)).unsqueeze(-1)
            #     we2_s = (torch.softmax(tmp_we2/tmp_we2.max(), dim=1)).unsqueeze(-1)

            #     # content_loss = content_loss + torch.mean((geo_mx*we1 - geo_fx*we2).square().sum(-1) / self.numel[d-2])
            #     # content_loss = content_loss + torch.mean((((geo_mx*we1 - geo_fx*we2).square()+ 1e-6).log()*self.masks[str(d)]).sum(-1) / self.numel[d-2])
            #     # content_loss = content_loss + torch.mean(((geo_mx-geo_fx).square()).sum(-1) / self.numel[d-2])
            #     # content_loss = content_loss - torch.mean( (torch.exp(-(geo_mx*we1-geo_fx*we2).square())*self.masks[str(d)]).sum(-1)/self.numel[d-2])
                
            #     # geo_mx = geo_mx * we1_s
            #     # geo_fx = geo_fx * we2_s
            #     cos_sim = F.cosine_similarity(geo_mx, geo_fx, dim=-1)

            #     # cos_sim = torch.sum(geo_mx * geo_fx, dim=-1) / (geo_mx.square().sum(-1).sqrt() * geo_fx.square().sum(-1).sqrt())
            #     # content_loss = content_loss + torch.mean(torch.exp(1-cos_sim))-1
            #     # content_loss = content_loss + torch.mean(torch.log(1 - cos_sim + 1e-6))
            #     content_loss = content_loss + torch.exp(1-cos_sim).mean()

            #     # sim = (geo_mx-geo_fx)/(geo_mx+geo_fx+1e-10)
            #     # content_loss = content_loss + torch.mean(torch.exp(cos_sim.sum(-1)/self.numel[d-2]))

            #     # diff = (geo_mx-geo_fx).square()
            #     # content_loss = content_loss + torch.mean( (diff / diff.sum(-1)) / self.numel[d-2] )
            #     # content_loss = content_loss - torch.mean( (-(diff / diff.sum(-1))).exp() * self.masks[str(d)]  )
            
        return style_loss, content_loss

    def setPrototype(self,x,x_filename,stage):
        p = self.prototypes[stage][x_filename]
        prototype = (1-self.moment) * x + self.moment * p
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
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        return X.bmm(X.transpose(1,2)) / (c * n)
    
    def gram2(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = X.bmm(X.transpose(1,2))
        factor = X.square().sum(-1,keepdims=True).sqrt()  # b, c
        norm = factor.bmm(factor.transpose(1,2))
        return gram / norm

    def style_loss(self, moved_prototype, prototype, y_type):
        center_loss = torch.square(self.gram(moved_prototype) - self.gram(prototype[str(y_type)])).mean()
        return center_loss

    def mse_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def modlity_loss(self, source_ft, source_file, source_pros):
        source_pro = source_pros[source_file].unsqueeze(0)
        # loss = torch.pow((source_ft-source_pro),2).mean()
        # loss = F.cosine_similarity(source_ft,source_pro,1).mean()
        loss = (self.gram(source_ft) - self.gram(source_pro)).square().mean()
        return loss
    
    def geo_loss(self, X, d):
        # cosine dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = torch.bmm(X.transpose(1,2), X) / c
        # return self.masks[str(d)] * gram
        return torch.spares.mm(gram,self.masks[str(d)])
    
    def geo_loss2(self, X, d):
        # Euc dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n)).unsqueeze(-1)
        gram = (X - X.transpose(-2,-1))
        # return self.masks[str(d)] * gram
        return gram * self.masks[str(d)].to_dense()
    
    def geo_loss3(self,X,d): # 3 Dim
        neigh_feat = X.unfold(2,self.k_size[d],1).unfold(3,self.k_size[d],1).unfold(4,self.k_size[d],1).flatten(-3) # B, C, Z//k_size+1, Z//k_size+1, Z//k_size+1, neigh_num
        neigh_feat = neigh_feat - neigh_feat[:,:,:,:,:,self.k_size[d]**3//2].unsqueeze(-1) # 减去邻域中心
        neigh_feat = neigh_feat.reshape(X.shape[0], X.shape[1], -1, neigh_feat.shape[-1]) # B, C, Z//k_size+1 * Z//k_size+1 * Z//k_size+1, neigh_num
        
        return neigh_feat
 
class Decoder(nn.Module):
    # 采用风格
    def __init__(self, out_ch=1, stage_num=5, conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Decoder,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = True

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool
        self.up_sample = up_sample

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        decode_blocks = [
            DoubleConv(2*16*base_ch, 16*base_ch, 8*base_ch),
            DoubleConv(3*8*base_ch, 8*base_ch, 4*base_ch),
            DoubleConv(3*4*base_ch, 4*base_ch, 2*base_ch),
            DoubleConv(3*2*base_ch, 2*base_ch, base_ch),
            DoubleConv(3*base_ch, base_ch, out_ch)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(3*8*base_ch, 3*8*base_ch, 2,2),
                nn.ConvTranspose3d(3*4*base_ch, 3*4*base_ch, 2,2),
                nn.ConvTranspose3d(3*2*base_ch, 3*2*base_ch, 2,2),
                nn.ConvTranspose3d(3*base_ch, 3*base_ch, 2,2),
            ]

        self.decode_blocks = nn.ModuleList(decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

    def forward(self, skip_m_proto, skip_f_proto, skip_m_content, skip_f_content):
        d_x = torch.concat([skip_m_content[-1], skip_f_proto[-1]], dim=1)
        d_x = self.decode_blocks[0](d_x)
        for u in range(1,self.stage_num):
            # x = torch.concat([d_x, skip_content_moving[-(u+1)], self.getPrototypes(y_filename,-(u+1))], dim=1)
            # d_x = torch.concat([d_x, skip_content_moving[-(u+1)], skip_proto_fixed[-(u+1)]], dim=1)
            d_x = torch.concat([d_x, skip_m_content[-(u+1)], skip_f_proto[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        return d_x

class Decoder2(Decoder):
    # 采用风格
    def __init__(self, out_ch=1, stage_num=5, conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Decoder2,self).__init__(out_ch, stage_num, conv_pool, pool_args, up_sample, up_sample_args)

        # 网络参数
        base_ch = 32

        decode_blocks = [
            DoubleConv(2*16*base_ch, 16*base_ch, 8*base_ch),
            DoubleConv(3*8*base_ch, 8*base_ch, 4*base_ch),
            DoubleConv(3*4*base_ch, 4*base_ch, 2*base_ch),
            DoubleConv(2*2*base_ch, 2*base_ch, base_ch),
            DoubleConv(2*base_ch, base_ch, out_ch)
        ]
        self.decode_blocks = nn.ModuleList(decode_blocks)

    def forward(self, skip_m_proto, skip_f_proto, skip_m_content, skip_f_content):
        d_x = torch.concat([skip_m_content[-1], skip_f_proto[-1]], dim=1)
        d_x = self.decode_blocks[0](d_x)
        for u in range(1,self.stage_num):
            # x = torch.concat([d_x, skip_content_moving[-(u+1)], self.getPrototypes(y_filename,-(u+1))], dim=1)
            # d_x = torch.concat([d_x, skip_content_moving[-(u+1)], skip_proto_fixed[-(u+1)]], dim=1)
            if u < self.stage_num-2:
                d_x = torch.concat([d_x, skip_m_content[-(u+1)], skip_f_proto[-(u+1)]], dim=1)
            else:
                d_x = torch.concat([d_x, skip_m_content[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        return d_x


class ModalEncoder(nn.Module):
    # 采用风格
    def __init__(self,inshape, types, in_ch=1, out_ch=1, stage_num=5,  conv_pool=False, pool_args=None,  freeze=False):
        super(ModalEncoder, self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = False

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
        else:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]

        self.encode_blocks = nn.ModuleList(encode_blocks) # 模态

        # self.contra_fn = ContrastiveLoss(1)
        self.triplet_fn = nn.TripletMarginLoss(6, p=2)
        self.infonce = InfoNCE(negative_mode='paired')

        input = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        prototypes = []
        self.masks = {}
        self.numel = []

        for d in range(self.stage_num):
            prototype = {}
            input = self.encode_blocks[d](input)
            for k in types:
                # prototype[k] = nn.Parameter(torch.rand_like(input.squeeze(0))) # using feature map as ptototype
                prototype[k] = nn.Parameter(normalize(torch.rand((input.shape[1] * input.shape[1]))), requires_grad=False) # using gram as prototype
                # prototype[k] = torch.rand((input.shape[1], input.shape[1]))
            prototypes.append(nn.ParameterDict(prototype))
            # prototypes.append(prototype[k])
        self.prototypes = nn.ParameterList(prototypes)
        # self.register_buffer('prototypes', prototypes)

        for d in range(self.stage_num):
            if freeze:
                for _, p in self.encode_blocks[d].named_parameters():
                    p.requires_grad = False
            
            if freeze_prototypes:
                for _, p in self.prototypes[d].items():
                    p.requires_grad = False

    def set_input(self, moving, fixed, moving_filename, fixed_filename):
        self.moving = moving
        self.fixed = fixed
        self.m_type = moving_filename
        self.f_type = fixed_filename

    def infer(self, movings, fixeds):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = movings
        y = fixeds

        skip_proto_moving =[]
        skip_proto_fixed =[]

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            
            skip_proto_moving.append(x)
            skip_proto_fixed.append(y)

        return  skip_proto_moving, skip_proto_fixed

    def forward1(self):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = self.moving
        y = self.fixed
        bs = x.shape[0]
        x_type = self.m_type
        y_type = self.f_type

        prototype_loss = torch.tensor(0)
        contra_loss = torch.tensor(0)

        self.skip_proto_moving =[]
        self.skip_proto_fixed =[]

        count = 0

        fixed_gram = []
        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            
            self.skip_proto_moving.append(x)
            self.skip_proto_fixed.append(y)
            
            x_gram = self.gram3(x)
            y_gram = self.gram3(y)

            fixed_gram.append(y_gram.detach())

            if d >= 0:
                count = count + 1
                # prototype_loss = prototype_loss + (x_gram - self.getPrototypes(x_type, d) +1e-12).square().sqrt().mean() \
                #             + (y_gram - self.getPrototypes(y_type, d) +1e-12).square().sqrt().mean()                
                # contra_loss = contra_loss + self.contra_fn(x_gram, y_gram, 1) # 对比损失
                # contra_loss  = contra_loss + self.triplet_fn(x_gram, self.getPrototypes(x_type, d), self.getPrototypes(y_type, d)) \
                #                             + self.triplet_fn(y_gram, self.getPrototypes(y_type, d), self.getPrototypes(x_type, d)) # 三元组损失

                # 正则化的（新）
                # prototype_loss = prototype_loss + torch.pairwise_distance(x_gram, self.getPrototypes(m_type, d)) \
                #                                 + torch.pairwise_distance(y_gram, self.getPrototypes(y_type, d))
                # contra_loss  = contra_loss + self.triplet_fn(x_gram, self.getPrototypes(x_type, d), self.getPrototypesNeg(x_type, d)) \
                #                             + self.triplet_fn(y_gram, self.getPrototypes(y_type, d), self.getPrototypesNeg(y_type, d)) # 三元组损失
                
                # 余弦距离
                # prototype_loss = prototype_loss + (1- F.cosine_similarity(x_gram, self.getPrototypes(x_type,d)))  \
                #                                 + (1- F.cosine_similarity(y_gram, self.getPrototypes(y_type,d))) 
                contra_loss = contra_loss + InfoNCE2(x_gram, self.getPrototypes(x_type, d), self.getPrototypesNeg(x_type, d), 0.07)
                contra_loss = contra_loss + InfoNCE2(y_gram, self.getPrototypes(y_type, d), self.getPrototypesNeg(y_type, d), 0.07)
                
                # infonce only
                # contra_loss = contra_loss + InfoNCE2(x_gram, self.getPrototypes(x_type, d), self.getPrototypesNeg(x_type, d), 0.07)
                # contra_loss = contra_loss + InfoNCE2(y_gram, self.getPrototypes(y_type, d), self.getPrototypesNeg(y_type, d), 0.07)
                
                self.setPrototype(x_gram.detach(), x_type, d)
                self.setPrototype(y_gram.detach(), y_type, d)
        
        self.fixed_gram = fixed_gram
        self.proto_loss = prototype_loss / count
        self.contra_loss = contra_loss /count

        return  self.skip_proto_moving, self.skip_proto_fixed
    
    def forward2(self, x_recon):
        # encoder again
        warped_x = x_recon

        style_loss = torch.tensor(0)
        count = 0

        for d in range(self.stage_num):
            warped_x = self.encode_blocks[d](warped_x)
            # style_loss = style_loss + (self.gram2(warped_x) - self.getPrototypes(y_filename, d)).square().mean()
            # style_loss = style_loss + (self.gram(warped_x) - self.gram(skip_proto_fixed[d])).square().mean()
            if d >= 0:
                count = count + 1
                # style_loss = style_loss + (self.gram3(warped_x) - self.getPrototypes(self.f_type, d) + 1e-12).square().sqrt().mean() # grad欧式
                # style_loss = style_loss + torch.pairwise_distance(self.gram3(warped_x), self.getPrototypes(self.f_type, d)) # grad欧式(新)
                # style_loss = style_loss + self.triplet_fn(self.gram3(warped_x), self.getPrototypes(self.f_type, d), self.getPrototypesNeg(self.f_type, d))
                # style_loss = style_loss + (self.gram3(warped_x) - self.fixed_gram[d] + 1e-12).square().sqrt().mean()
                # style_loss = style_loss + F.cosine_similarity(self.gram3(warped_x).flatten(1) , self.fixed_gram[d].flatten(1))
                # style_loss = style_loss + (1- F.cosine_similarity(self.gram3(warped_x), self.getPrototypes(self.f_type,d))) # grad余弦
                style_loss = style_loss + (1- F.cosine_similarity(self.gram3(warped_x), self.fixed_gram[d])) # grad余弦
                # style_loss = style_loss + InfoNCE2(self.gram3(warped_x), self.getPrototypes(self.f_type, d), self.getPrototypesNeg(self.f_type, d), 0.07) # grad余弦(new)
        return style_loss / count

    def setPrototype(self, x, x_filename, stage):
        for B, filename in enumerate(x_filename):
            p = self.prototypes[stage][filename]
            prototype = (1-self.moment) * x[B] + self.moment * p
            self.prototypes[stage][filename] = normalize(prototype.detach())
    
    def getPrototypes(self,x_filename,stage):
        prototypes = []
        for filename in x_filename:
            prototypes.append(self.prototypes[stage][str(filename)])
        return torch.stack(prototypes).detach()

    def getPrototypesNeg(self,x_filename,stage):
        neg_prototypes = []
        for filename in x_filename:
            proto_copy = self.prototypes[stage].copy()
            proto_copy.pop(str(filename))
        return torch.stack(list(proto_copy.values())).detach()

    def gram(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        return X.bmm(X.transpose(1,2)) / (c * n)
    
    def gram2(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = X.bmm(X.transpose(1,2))
        factor = X.square().sum(-1,keepdims=True).sqrt()  # b, c
        norm = factor.bmm(factor.transpose(1,2))
        return gram / norm

    def gram3(self, X):
        X = X.flatten(2) # B,C,N
        gram = X.bmm(X.transpose(1,2)).flatten(1) # B,C*C
        return F.normalize(gram, dim=1)

    def style_loss(self, moved_prototype, prototype, y_type):
        center_loss = torch.square(self.gram(moved_prototype) - self.gram(prototype[str(y_type)])).mean()
        return center_loss

    def modlity_loss(self, source_ft, source_file, source_pros):
        source_pro = source_pros[source_file].unsqueeze(0)
        # loss = torch.pow((source_ft-source_pro),2).mean()
        # loss = F.cosine_similarity(source_ft,source_pro,1).mean()
        loss = (self.gram(source_ft) - self.gram(source_pro)).square().mean()
        return loss
   

class MorphEncoder(nn.Module):
    # 采用风格
    def __init__(self,inshape, in_ch=1, out_ch=1, stage_num=5, conv_pool=False, pool_args=None,  freeze=False):
        super(MorphEncoder, self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = True

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]

        else:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]

        self.encode_blocks = nn.ModuleList(encode_blocks) # 形态

        input = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        self.masks = {}
        self.numel = []
        self.unfold = []
        # self.k_size = torch.tensor([[9,9,9], [5,5,5], [3,3,3], [2,3,3], [1,3,3]])
        self.k_size = torch.tensor([[3,3,3], [3,3,3], [3,3,3], [2,3,3], [1,3,3]])
        self.ss = torch.tensor([[3,3,3], [3,3,3], [3,3,3], [2,3,3], [1,3,3]])
        self.center = self.k_size.div(2, rounding_mode='trunc')
        self.center = self.center[:,0] * self.k_size[:,1] * self.k_size[:,2] \
                        + self.center[:,1] * self.k_size[:,2] + self.center[:,2]
        for d in range(len(self.k_size)):
            self.unfold.append(unfoldNd.UnfoldNd(kernel_size=self.k_size[d].tolist(), stride=1))
        # for d in range(self.stage_num):
        #     input = self.encode_blocks[d](input)
        #     gap = self.k_size[d] // 2
        #     if d > 2:
        #         mask = []
        #         for z in range(input.shape[2]):
        #             for h in range(input.shape[3]):
        #                 for w  in range(input.shape[4]):
        #                     tmp_mask = torch.zeros_like(input[0,0],dtype=torch.bool)
        #                     tmp_mask[max(z-gap,0):z+gap+1,max(h-gap,0):h+gap+1,max(w-gap,0):w+gap+1] = True
        #                     mask.append(tmp_mask.flatten())

        #         mask = (torch.stack(mask).unsqueeze(0) * ~torch.eye(input[0,0].numel(),dtype=bool)).cuda()
        #         mask_idx = torch.nonzero(mask).T  # 这里需要转置一下
        #         mask_data = mask[mask_idx[0], mask_idx[1], mask_idx[2]]
        #         coo_mask = torch.sparse_coo_tensor(mask_idx, mask_data, mask.shape)

        #         self.masks[str(d)] = coo_mask
        #         # self.numel.append(self.masks[str(d)].sum(-1).byte())

        for d in range(self.stage_num):
            if freeze:
                for _, p in self.encode_blocks[d].named_parameters():
                    p.requires_grad = False

    def set_input(self, moving, fixed, moving_filename, fixed_filename):
        self.moving = moving
        self.fixed = fixed
        self.m_type = moving_filename
        self.f_type = fixed_filename

    def infer(self, movings, fixeds):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = movings
        y = fixeds

        skip_content_moving = []
        skip_content_fixed = []

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            
            skip_content_moving.append(x)
            skip_content_fixed.append(y)

        return  skip_content_moving, skip_content_fixed

    def forward1(self):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = self.moving
        y = self.fixed

        self.skip_content_moving = []
        self.skip_content_fixed = []

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            
            self.skip_content_moving.append(x)
            self.skip_content_fixed.append(y)

        return  self.skip_content_moving, self.skip_content_fixed

    
    def forward2(self, x_recon):
        # encoder again
        warped_x = x_recon

        content_loss = torch.tensor(0)

        for d in range(self.stage_num):

            warped_x = self.encode_blocks[d](warped_x)
            
            if d >= 2:
                geo_mx = self.geo_loss3(warped_x, d) # m->f, B,C,ZHW,ZHW
                geo_fx = self.geo_loss3(self.skip_content_moving[d], d)
                cos_sim = F.cosine_similarity(geo_mx, geo_fx, dim=-1)
                content_loss = content_loss + torch.exp(1-cos_sim).mean()
            
        return content_loss
    
    def geo_loss(self, X, d):
        # cosine dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = torch.bmm(X.transpose(1,2), X) / c
        # return self.masks[str(d)] * gram
        return torch.spares.mm(gram,self.masks[str(d)])
    
    def geo_loss2(self, X, d):
        # Euc dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n)).unsqueeze(-1)
        gram = (X - X.transpose(-2,-1))
        # return self.masks[str(d)] * gram
        return gram * self.masks[str(d)].to_dense()
    
    # def geo_loss3(self,X,d): # 3 Dim
    #     neigh_feat = X.unfold(2,self.k_size[d],1).unfold(3,self.k_size[d],1).unfold(4,self.k_size[d],1).flatten(-3) # B, C, Z//k_size+1, Z//k_size+1, Z//k_size+1, neigh_num
    #     neigh_feat = neigh_feat - neigh_feat[:,:,:,:,:,self.k_size[d]**3//2].unsqueeze(-1) # 减去邻域中心
    #     neigh_feat = neigh_feat.reshape(X.shape[0], X.shape[1], -1, neigh_feat.shape[-1]) # B, C, Z//k_size+1 * Z//k_size+1 * Z//k_size+1, neigh_num
        
    #     return neigh_feat

    def geo_loss3(self, X, d): # 3 Dim, 用于维度非等长张量
        neigh_feat = X.unfold(2,self.k_size[d][0],1).unfold(3,self.k_size[d][1],1).unfold(4,self.k_size[d][2],1).flatten(-3) # B, C, Z//k_size+1, Z//k_size+1, Z//k_size+1, neigh_num
        neigh_feat = neigh_feat - neigh_feat[:,:,:,:,:,self.center[d]].unsqueeze(-1) # 减去邻域中心
        neigh_feat = neigh_feat.reshape(X.shape[0], X.shape[1], -1, neigh_feat.shape[-1]) # B, C, Z//k_size+1 * Z//k_size+1 * Z//k_size+1, neigh_num
        return neigh_feat

    def geo_loss4(self, X, d):
        B, C = X.shape[0], X.shape[1]
        unfolded = self.unfold[d](X)
        num_block = unfolded.shape[-1]
        return unfolded.reshape(B, C, -1, num_block)

class Decoder2(nn.Module):
    # 采用风格
    def __init__(self, out_ch=1, stage_num=5, conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Decoder2,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = True

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool
        self.up_sample = up_sample

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        decode_blocks = [
            DoubleConv(2*16*base_ch, 16*base_ch, 8*base_ch),
            DoubleConv(3*8*base_ch, 8*base_ch, 4*base_ch),
            DoubleConv(3*4*base_ch, 4*base_ch, 2*base_ch),
            DoubleConv(2*2*base_ch, 2*base_ch, base_ch),
            DoubleConv(2*base_ch, base_ch, out_ch)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(3*8*base_ch, 3*8*base_ch, 2,2),
                nn.ConvTranspose3d(3*4*base_ch, 3*4*base_ch, 2,2),
                nn.ConvTranspose3d(2*2*base_ch, 2*2*base_ch, 2,2),
                nn.ConvTranspose3d(2*base_ch, 2*base_ch, 2,2),
            ]

        self.decode_blocks = nn.ModuleList(decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

    def forward(self, skip_m_proto, skip_f_proto, skip_m_content, skip_f_content):
    
        d_x = torch.concat([skip_m_content[-1], skip_f_proto[-1]],dim=1)
        d_x = self.decode_blocks[0](d_x)
        for u in range(1,3):
            # x = torch.concat([d_x, skip_content_moving[-(u+1)], self.getPrototypes(y_filename,-(u+1))], dim=1)
            # d_x = torch.concat([d_x, skip_content_moving[-(u+1)], skip_proto_fixed[-(u+1)]], dim=1)
            d_x = torch.concat([d_x, skip_m_content[-(u+1)], skip_f_proto[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        for u in range(3,self.stage_num):
            # x = torch.concat([d_x, skip_content_moving[-(u+1)], self.getPrototypes(y_filename,-(u+1))], dim=1)
            # d_x = torch.concat([d_x, skip_content_moving[-(u+1)], skip_proto_fixed[-(u+1)]], dim=1)
            d_x = torch.concat([d_x, skip_m_content[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        return d_x
    
if __name__ == '__main__':
    encoder = Encoder(inshape=(64,64,64))
    inputs = torch.rand((2,1,64,64,64))
    kip_proto_moving, skip_proto_fixed, skip_content_moving, skip_content_fixed = encoder.forward1(inputs)
    style_loss, content_loss = encoder.forward2(inputs)