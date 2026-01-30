import torch
import torch.nn.functional as F
import numpy as np
import math

from torch.nn import ConstantPad3d, ReplicationPad3d


class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred, reduction='mean'):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        if reduction == 'mean':
            return -torch.mean(cc)
        elif reduction == 'batch':
            return -torch.mean(cc.flatten(1), dim=1)
        else:
            return -cc


class MSE(torch.nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super(MSE,self).__init__()

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


# class Dice:
#     """
#     N-D dice for segmentation
#     """

#     def loss(self, y_true, y_pred):
#         ndims = len(list(y_pred.size())) - 2
#         vol_axes = list(range(2, ndims + 2))
#         top = 2 * (y_true * y_pred).sum(dim=vol_axes)
#         bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
#         dice = torch.mean(top / bottom)
#         return -dice

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        c = output1.shape[1]
        # output1和output2为两个向量，label表示两向量是否是同一类，同一类为0,不同类为1
        euclidean_distance = F.pairwise_distance(output1, output2) / c
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
        return loss_contrastive

class ContrastiveLoss2(torch.nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class MINDSSCLoss(torch.nn.Module):
    """
    Modality-Independent Neighbourhood Descriptor Dissimilarity Loss for Image Registration
    References: https://link.springer.com/chapter/10.1007/978-3-642-40811-3_24

    Args:
        radius (int): radius of self-similarity context.
        dilation (int): the dilation of neighbourhood patches.
        penalty (str): the penalty mode of mind dissimilarity loss.
    """
    def __init__(
        self,
        radius: int = 2,
        dilation: int = 2,
        penalty: str = 'l2',
    ) -> None:
        super().__init__()
        self.kernel_size = radius * 2 + 1
        self.dilation = dilation
        self.radius = radius
        self.penalty = penalty
        self.mshift1, self.mshift2, self.rpad1, self.rpad2 = self.build_kernels(
        )

    def build_kernels(self):
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                                          [1, 1, 2], [2, 1, 1], [1, 2,
                                                                 1]]).long()

        # squared distances
        dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask, square distance equals 2
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        # self-similarity context: 12 elements
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 +
                         idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift1.requires_grad = False

        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1,
                                                           1).view(-1,
                                                                   3)[mask, :]
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 +
                         idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        mshift2.requires_grad = False

        # maintain the output size
        rpad1 = ReplicationPad3d(self.dilation)
        rpad2 = ReplicationPad3d(self.radius)
        return mshift1, mshift2, rpad1, rpad2

    def mind(self, img: torch.Tensor) -> torch.Tensor:
        mshift1 = self.mshift1.to(img)
        mshift2 = self.mshift2.to(img)
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2(
            (F.conv3d(self.rpad1(img), mshift1, dilation=self.dilation) -
             F.conv3d(self.rpad1(img), mshift2, dilation=self.dilation))**2),
                           self.kernel_size,
                           stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var,
                               mind_var.mean() * 0.001,
                               mind_var.mean() * 1000)
        mind = torch.div(mind, mind_var)
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:,
                    torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(
                    ), :, :, :]

        return mind

    def forward(self, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Compute the MIND-SSC loss.

        Args:
            source: source image, tensor of shape [BNHWD].
            target: target image, tensor fo shape [BNHWD].
        """
        assert source.shape == target.shape, 'input and target must have the same shape.'
        if self.penalty == 'l1':
            mind_loss = torch.abs(self.mind(source) - self.mind(target))
        elif self.penalty == 'l2':
            mind_loss = torch.square(self.mind(source) - self.mind(target))
        else:
            raise ValueError(
                f'Unsupported penalty mode: {self.penalty}, available modes are l1 and l2.'
            )

        return torch.mean(mind_loss)  # the batch and channel average

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(radius={self.radius},'
                     f'dilation={self.dilation},'
                     f'penalty=\'{self.penalty}\')')
        return repr_str


def reg_loss(fixed, flows):
    # sim_loss = pearson_correlation(fixed, warped)
    # Regularize all flows
    if len(fixed.size()) == 4: #(N, C, H, W)
        reg_loss = sum([regularize_loss(flow) for flow in flows])
    else:
        reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    
    return  reg_loss 

def reg_loss2(fixed, flows):
    # sim_loss = pearson_correlation(fixed, warped)
    # Regularize all flows
    if len(fixed.size()) == 4: #(N, C, H, W)
        reg_loss = [regularize_loss(flow) for flow in flows]
    else:
        reg_loss = [regularize_loss_3d(flow) for flow in flows]
    
    return  reg_loss 

def reg_loss1(flow):
    return flow.flatten(1).pow(2).mean(1).mean()

def regularize_loss_3d(flow):
    """
    flow has shape (batch, 3, 512, 521, 512)
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    d = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0

def regularize_loss(flow):
    """
    flow has shape (batch, 2, 521, 512)
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:] - flow[..., :-1]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0

def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1) ** 2)
    var2 = torch.mean((flatten_warped - mean2) ** 2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    return raw_loss
 
def Center_loss(source_ft,source_file,source_pros):
    """
    center loss 
    source_fts: feature of the moving image,
    source_pro: now prototypes
    """
    source_pro = source_pros[source_file]
    center_loss = torch.mean(torch.square((source_ft-source_pro)))

    return center_loss

def triplet_loss(source_ft,source_file,prototypes,scalar):
    dif = 0
    for k,v in prototypes:
        if k==source_file:
            same = torch.sqrt(source_ft-v)
        else:
            dif+=torch.sqrt(source_ft-v)
    triplrt_loss = same + dif
    return triplrt_loss

def pdist_squared(x: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise squared euclidean distance of input coordinates.

    Args:
        x: input coordinates, input shape should be (1, dim, #input points)
    Returns:
        dist: pairwise distance matrix, (#input points, #input points)
    """
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


