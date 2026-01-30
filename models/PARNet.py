from turtle import forward

from torch import device
from models.UNet2 import *
from models.recursive_cascade_networks import RecursiveCascadeNetwork
from .modelio import LoadableModel, store_config_args
from config import args
class PARNet(LoadableModel):
    @store_config_args
    def __init__(self,inshape,in_channel,out_channel,n_cascades=1):
        super(PARNet,self).__init__()
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )
        self.unet_model = UNet2(inshape=inshape).to(device)
        
        self.cascade_networks = RecursiveCascadeNetwork(n_cascades)
        self.trainable_params = []
        self.trainable_params +=self.unet_model.parameters()
        for submodel in self.cascade_networks.stems:
            self.trainable_params += list(submodel.parameters())
        self.trainable_params += list(self.cascade_networks.reconstruction.parameters())
        
    def forward(self,moving,fixed,moving_filename,fixed_filename):

        warped_init,source_ft,ini_reg_loss,center_loss = self.unet_model(moving,moving_filename,fixed,fixed_filename)
        
        warped ,flows = self.cascade_networks(fixed,warped_init)
        
        return warped_init, warped ,flows,ini_reg_loss,center_loss
