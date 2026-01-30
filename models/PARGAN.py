# Affine->PANet->VTN
# ++from turtle import forward

from models.UNet7 import Reg, Trans
from models.base_networks import VTNAffineStem, VTN
from models.recursive_cascade_networks import RecursiveCascadeNetwork
from .modelio import LoadableModel, store_config_args
from config import args

class PARGAN(nn.Module):
    def __init__(self,inshape,device,n_cascades=1):
        super(PARGAN,self).__init__()
        self.stems = []
        self.device = device
        self.reg = Reg(inshape=inshape).to(device)
        self.trans = Trans(inshape=inshape).to(device)
        
        
    def forward(self,moving,fixed,moving_filename,fixed_filename):
        flows = []
        results = []

        warped_pa, flow, prototype_loss, consistency_loss, style_loss = self.unet_model(moving,fixed,moving_filename,fixed_filename)
        flows.append(flow)
        results.append(warped_pa)
        flow, id_loss = self.affine(results[-1],fixed)
        flows.append(flow)
        results.append(self.reconstruction(results[-1], flows[-1]))
        
        for submodel in self.stems:
            flows.append(submodel(results[-1], fixed))
            results.append(self.reconstruction(results[-1], flows[-1]))

        return results, flows, id_loss, prototype_loss, consistency_loss, style_loss 

