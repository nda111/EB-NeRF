import os
import itertools
from datetime import datetime

from omegaconf import OmegaConf
from argparse import ArgumentParser

from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import imageio

import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.backends.cudnn

import models
import data
import utils

#region Configuration
parser = ArgumentParser('NeRF Blending')
add = parser.add_argument
add('--config', '-c', type=str, nargs='*', default=[])

add('--device', type=str)
add('--seed', type=int)
add('--deterministic', action='store_true')

add('--split', type=str)
add('--no_shuffle', action='store_true')
add('--n_rays', type=int)

add('--expname', '-n', type=str)
add('--architecture', '-a', type=str, choices=models.arch_names)

add('--epochs', type=int)
add('--learning_rate', type=float)

add('--eval_period', type=int)
add('--render_period', type=int)

del add
CFG = parser.parse_args()
CFG = {key: val for key, val in vars(CFG).items() if val is not None}

OmegaConf.register_new_resolver('auto_device', resolver=lambda: 'cuda:0' if torch.cuda.is_available() else 'cpu')
OmegaConf.register_new_resolver('today', resolver=lambda: datetime.today().strftime('%y%m%d'))
OmegaConf.register_new_resolver('now', resolver=lambda: datetime.today().strftime('%H%M%S'))
CFG = OmegaConf.create(CFG)

cfg_list = []
for cfg_name in ['default'] + CFG.config:
    cfg_name = os.path.join('./config', cfg_name.replace('.', os.sep))
    if not cfg_name.lower().endswith('.yaml'):
        cfg_name += '.yaml'
    cfg_list.append(OmegaConf.load(cfg_name))
CFG = OmegaConf.merge(*cfg_list, CFG)

split_filename = f'./config/split/{CFG.split}'
if not split_filename.lower().endswith('.yaml'):
    split_filename += '.yaml'
SPLIT_INFO = OmegaConf.load(split_filename)
#endregion

#region Exp. directory, Logging
EXP_DIR = Path('./log').joinpath(CFG.expname)
CODE_DIR = EXP_DIR.joinpath('code')
CKPT_FILENAME = EXP_DIR.joinpath('ckpt.pth')
LOG_FILENAME = EXP_DIR.joinpath('train.log')

CODE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('TRAIN')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def _print(*args, sep=' ', cli=True, file=True):
    str_args = [str(arg) for arg in args]
    line = sep.join(str_args)
    if file:
        logger.info(line)
    if cli:
        print(line)

_print(CFG)

writer = SummaryWriter(EXP_DIR)
os.system(f'cp "{__file__}" "{CODE_DIR}"')
os.system(f'cp "{split_filename}" "{CODE_DIR}/split.yaml"')
#endregion

#region Device, Reproduction
DEVICE = torch.device(CFG.device.lower())
SEED = torch.default_generator.initial_seed() if CFG.seed is None else CFG.seed
SEED = SEED % 2**32
DETERMINISTIC = CFG.deterministic
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.set_device(DEVICE)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = not DETERMINISTIC
    torch.backends.cudnn.deterministic = DETERMINISTIC
#endregion

#region Dataset
def load_datasets(split: str):
    kwargs = dict(
        device=DEVICE,
        split=split,
        n_rays=CFG.n_rays,
        shuffle=(not CFG.no_shuffle) and (split == 'train'),
    )
    for info in SPLIT_INFO[f'novel_{split}']:
        data_name, scene_name = info.split('.')
        yield data.RayDataset(data_name, scene_name, **kwargs)

trainset_cycle = itertools.cycle(load_datasets('train'))
evalset_cycle = itertools.cycle(load_datasets('val'))
#endregion

#region Experts, Networks, Optimizer
load_expert_fn = {
    models.generator.VANILLA_NERF: lambda data_name, scene_name: 
        torch.load(f'nerf/logs/{data_name}/e100k/{scene_name}')['network_fine_state_dict'],
    # models.generator.POINT_NERF: '',  # TODO:
    # models.generator.INSTANT_NGP: '',
}[CFG.architecture]
experts_info = [info.split('.') for info in SPLIT_INFO['experts']]
experts = [load_expert_fn(data_name, scene_name) for data_name, scene_name in experts_info]

pool = models.blending.pool.TensorBlendingPool(experts)
serializer = pool.serializer
blender = models.blending.Blender(expert_dim=pool.expert_dim, num_experts=pool.num_experts).to(DEVICE)

generator_kwargs = models.generator.arch_kwargs_map[CFG.architecture]
generator = models.generator.Generator(arch=CFG.architecture, **generator_kwargs).to(DEVICE).eval()
optimizer = optim.Adam(params=generator.parameters(), lr=CFG.learning_rate)
#endregion

#region Training Loop
with tqdm(range(1, CFG.epochs + 1), desc='EPOCH', position=1, leave=False) as epoch_bar:
    for epoch in epoch_bar:
        _print(f'[EPOCH {epoch}]', cli=False)
        check_period = lambda p: ((epoch % p == 1) or (epoch == CFG.epochs))
        
        # train blender
        blender.train()
        train_rays, train_rgbs = next(trainset_cycle)
        
        # blending
        weights = blender(train_rays)  # ..........................................| 1. Inference blending weights.
        expert_inf = pool(weights)  # tensor form with grads ......................| 2. Actually blend the experts.
        generator.load_state_dict(serializer.deserialize(expert_inf)) # ...........| 3. Apply to the network.

        output = generator(train_rays)  # .........................................| 4. TODO: Synthesize the novel-view.
        generator_loss = nn.functional.mse_loss(output, train_rgbs)  # ............| 5. TODO: Calculate the loss as like NeRF.
        generator_loss.backward()  # ..............................................| 6. Calculate the generator gradient.
        
        # gradient stitching
        current_learning_rate = optimizer.param_groups[0]['lr']  # ................| 7. Make a pseudo ground truth.
        expert_inf_updated = generator.state_dict()
        expert_inf_grad = serializer(expert_inf_updated, acquire_grad=True)
        generator.zero_grad()  # ..................................................| Discard redundant gradients.
        expert_inf_target = expert_inf.clone().detach()
        expert_inf_target -= expert_inf_grad * current_learning_rate  
        blender_loss = nn.functional.mse_loss(expert_inf, expert_inf_target)  # ...| 8. Calculate the blender MSE.
        _print(f'{generator_loss=:.5E}', cli=False)
        _print(f'{blender_loss=:.5E}', cli=False)
        
        optimizer.zero_grad()
        blender_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            # eval blender if it is the epoch to
            if check_period(CFG.eval_period):
                blender.eval()
                eval_rays, eval_rgbs = next(evalset_cycle)
                
                weights = blender(eval_rays)
                expert_inf = pool(weights)
                generator.load_state_dict(serializer.deserialize(expert_inf)) 
                
                output = generator(eval_rays)
                psnr       = utils.metrics.psnr(output, eval_rgbs)               # TODO:
                ssim       = utils.metrics.ssim(output, eval_rgbs)               # TODO:
                lpips_alex = utils.metrics.lpips(output, eval_rgbs, net='alex')  # TODO:
                lpips_vgg  = utils.metrics.lpips(output, eval_rgbs, net='vgg')   # TODO:
                epoch_bar.set_postfix_str(f'PSNR={psnr:.2f}, SSIM={ssim:.4f}')
                _print(f'PSNR={psnr:.2f}', cli=False)
                _print(f'SSIM={ssim:.5f}', cli=False)
                _print(f'LPIPS_alex={lpips_alex:.4f}', cli=False)
                _print(f'LPIPS_vgg={lpips_vgg:.4f}', cli=False)
            else:
                dataset = None

            # render demo video
            if check_period(CFG.render_period):
                if dataset is None:
                    eval_rays, eval_rgbs = next(evalset_cycle)
                # TODO:
            pass
#endregion
