from .edsr import EDSR
from .fsrcnn import FSRCNN
from .carn import CARN_M
from .qp_net import QPNET

def build_sr(model_name, input_channels, output_channels, num_channels, num_blocks, scale):
    if model_name == 'edsr':
        model = EDSR(input_channels=input_channels, output_channels=output_channels, num_channels=num_channels, upscale=scale)
    elif model_name == 'fsrcnn':
        model = FSRCNN(input_channels=3, output_channels=3, upscale=scale, d=num_channels, m=num_blocks)
    elif model_name == 'carn':
        model = CARN_M(in_nc=3, out_nc=3, nf=num_channels, scale=scale)
    else:
        raise NotImplementedError

    return model

def build_qpnet(num_layers, num_channels, output_dim, patch_size):
    model = QPNET(num_layers=num_layers, num_channels=num_channels, output_dim=output_dim, patch_size=patch_size)

    return model