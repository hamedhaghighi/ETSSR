from models.model import Net
from models.StreoSwinSR import Net as SSSR
from model_zoo.PASSRnet import PASSRnet
from model_zoo.VDSR import VDSR
from model_zoo.RDN import RDN
from model_zoo.RCAN import RCAN
from model_zoo.EDSR import EDSR
from model_zoo.SRresNet_SAM import _NetG_SAM

def model_selection(model_name, upscale_factor=2, H=None, W=None, C=None, w_size=None, device='cuda'):
    if 'mine' in model_name:
        net = Net(upscale_factor=upscale_factor, model=model_name, img_size=tuple([H, W]), input_channel=C, w_size=w_size)
    elif 'transformer' in model_name:
        net = SSSR(upscale_factor, img_size=tuple([H, W]), model=model_name, input_channel=C, w_size=w_size, embed_dim=64)
    elif model_name == 'PASSRnet':
        net = PASSRnet(upscale_factor)
    elif model_name == 'VDSR':
        net = VDSR(upscale_factor)
    elif model_name == 'RDN':
        net = RDN(upscale_factor)
    elif model_name == 'RCAN':
        net = RCAN(upscale_factor)
    elif model_name == 'EDSR':
        net = EDSR(upscale_factor)
    elif model_name == 'SAM':
        net = _NetG_SAM(upscale_factor)
    return net.to(device)
