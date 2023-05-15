import os, sys
if __name__ == '__main__':
    sys.path.append(os.path.abspath('.'))

from typing import Literal
import functools

import torch
from torch import nn


class Blender(nn.Module):
    __aggregation_fn_map = {
        None     : lambda x: x,
        'none'   : lambda x: x,
        'linear' : lambda x: x / torch.sum(dim=-1, keepdim=True),
        'softmax': functools.partial(torch.softmax, dim=-1),
    }
    
    def __init__(self, expert_dim: int, num_experts: int, 
                 transit_steps: int=3, W_last: int=1024, batch_groups: int=1, 
                 D=8, W=256, input_ch=3, input_ch_views=3, skips=[4], use_viewdirs=False, 
                 aggregation: Literal[None, 'none', 'linear', 'softmax']=None, **_):
        super(Blender, self).__init__()
        
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.transit_steps = transit_steps
        self.W_last = W_last
        self.batch_groups = batch_groups
        
        self.D, self.W = D, W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.aggregation = aggregation
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + 
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) 
             for i in range(D - 1)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            ### Implementation according to the official code release 
            # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
            out_size = W // 2
        else:
            out_size = W
            
        self.last_w_layer = nn.Linear(out_size, W_last)
        self.group_batches = lambda x: torch.stack([x[i::batch_groups][:x.size(0) // batch_groups] 
                                                    for i in range(batch_groups)], dim=0)
        self.symm_avg_pool = functools.partial(torch.mean, dim=1)
        
        dims = torch.linspace(W_last, expert_dim * num_experts, transit_steps + 1)
        dims = dims.round().int()
        self.output_linear = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.relu(h)
        
        h = torch.relu(self.last_w_layer(h))
        h = self.group_batches(h)
        h = self.symm_avg_pool(h)
        
        for layer in self.output_linear:
            h = torch.relu(layer(h))
            
        h = h.reshape(-1, self.num_experts, self.expert_dim).mean(dim=0)
        output = Blender.__aggregation_fn_map[self.aggregation](h)
        
        return output


if __name__ == '__main__':
    kwargs = dict(
        batch_groups=1,
        # expert_dim=595844,
        expert_dim=24,
        num_experts=5,
        W_last=1024,
        aggregation='softmax',
    )

    device = 'cuda'
    sample_x = torch.randn(10000, 6).to(device)
    net = Blender(**kwargs).to(device)

    assert net(sample_x).shape == (5, 24)
