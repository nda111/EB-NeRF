import os, sys
if __name__ == '__main__':
    sys.path.append(os.path.abspath('.'))
# sys.path.append(os.path.abspath('./point_nerf'))
sys.path.append(os.path.abspath('./instant_ngp'))

import torch
from torch import nn

from nerf.run_nerf_helpers import NeRF as VanillaNeRF
# from point_nerf.models.neural_points_volumetric_model import NeuralPointsVolumetricModel as PointNeRF
# from instant_ngp.nerf.network import NeRFNetwork as InstantNGP


VANILLA_NERF = 'vanilla_nerf'
# POINT_NERF = 'point_nerf'  # TODO:
# INSTANT_NGP = 'instant_npg'
arch_names = (
    VANILLA_NERF,
    # POINT_NERF,
    # INSTANT_NGP,
)

arch_kwargs_map = {
    VANILLA_NERF: torch.load('./models/kwargs_fine.pt'),
}


class Generator(nn.Module):
    __arch_map = {
        VANILLA_NERF: {
            'class': VanillaNeRF,
            'preproc': lambda x: (x,),
            'postproc': lambda _: _,
        },
        # POINT_NERF: {  # TODO:
        #     'class': PointNeRF,
        #     'preproc': lambda _: _,
        #     'postproc': lambda _: _,
        # },
        # INSTANT_NGP: {
        #     'class': InstantNGP,
        #     'preproc': lambda x: torch.split(x, [3, 3], dim=1),
        #     'postproc': lambda out: torch.cat([out[1], out[0].unsqueeze(-1)], dim=1),
        # }
    }
    
    def __init__(self, arch, *args, **kwargs):
        super(Generator, self).__init__()
        self.arch = arch
        self.args = args
        self.kwargs = kwargs
        
        if self.arch not in arch_names:
            raise RuntimeError(f'Invalid {arch=}. Please select one of {arch_names}.')
        
        arch = Generator.__arch_map[arch]
        self.network = arch['class'](*args, **kwargs)
        self.preproc = arch['preproc']
        self.postproc = arch['postproc']
        
    def forward(self, x):
        x = self.preproc(x)
        y = self.network(*x)
        return self.postproc(y)
    
    def __str__(self) -> str:
        return str(self.network)
    
    def __repr__(self):
        return repr(self.network)


def __test__(num_rays=16, cuda=0):
    shape_x = (num_rays, 6)
    shape_y = (num_rays, 4)
    sample_x = torch.randn(*shape_x).cuda(cuda)
    
    for arch in arch_names:
        print(arch, end=' ')
        net = Generator(arch=arch).eval().cuda(cuda)
        out = net(sample_x)
        print(out.shape)
        assert (out.shape == shape_y)
    

if __name__ == '__main__':
    __test__()
    print('PASSED:', __file__)
