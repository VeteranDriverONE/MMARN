# Affine->PANet->VTN
# ++from turtle import forward

from models.UNet4 import *
from models.base_networks import VTNAffineStem, VTN
from models.recursive_cascade_networks import RecursiveCascadeNetwork
from .modelio import LoadableModel, store_config_args
from config import args
class PARNet2(nn.Module):
    def __init__(self,inshape,device,n_cascades=1):
        super(PARNet2,self).__init__()
        self.stems = []
        self.device = device
        self.unet_model = UNet4(inshape=inshape).to(device)
        id_map = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(device)
        self.affine = VTNAffineStem(dim=len(inshape), im_size=inshape[0],id_map=id_map).to(device)
        for i in range(n_cascades):
            self.stems.append(VTN(dim=len(inshape), flow_multiplier=1.0 / n_cascades).to(device))
        self.reconstruction = SpatialTransformer(inshape).to(device)

        self.trainable_params = []
        self.trainable_params += list(self.affine.parameters())
        self.trainable_params += self.unet_model.parameters()
        for submodel in self.stems:
            self.trainable_params += list(submodel.parameters())
        self.trainable_params += list(self.reconstruction.parameters())
        
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

