import os, sys
if __name__ == '__main__':
    sys.path.append(os.path.abspath('.'))
    
from omegaconf import OmegaConf
from io import StringIO

import torch
from torch.utils.data import Dataset

from data.blender import BlenderDataset
from data.llff import LlffDataset


__DATAROOT = './nerf/data'
__CONFIGROOT = './nerf/configs'
LLFF_DATA = 'llff'
SYNTHETIC = 'synth'
__DATADIR_MAP = {
    LLFF_DATA: 'nerf_llff_data',
    SYNTHETIC: 'nerf_synthetic',
}

dataset_names = (LLFF_DATA, SYNTHETIC)
scenes_map = {data_name: sorted(os.listdir(os.path.join(__DATAROOT, __DATADIR_MAP[data_name])))
              for data_name in dataset_names}
if 'README.txt' in scenes_map[SYNTHETIC]:
    os.remove(os.path.join(__DATAROOT, __DATADIR_MAP[SYNTHETIC], 'README.txt'))
    scenes_map[SYNTHETIC].remove('README.txt')
scene_names = sorted(sum([scenes_map[dn] for dn in dataset_names], []))


def _get_datadir(dataset, scene):
    if dataset not in dataset_names:
        raise RuntimeError(f'Invalid {dataset=}. Please select one of {dataset_names}.')
    
    dataset_dir = os.path.join(__DATAROOT, __DATADIR_MAP[dataset])
    scenes = scenes_map[dataset]
    if isinstance(scene, int):
        scene = scenes[scene]
    
    if scene not in scenes:
        raise RuntimeError(f'Scene \'{scene}\' does not exist. Please select one of {scenes}.')
    
    return os.path.join(dataset_dir, scene)


def _get_config(scene):
    if scene not in scene_names:
        raise RuntimeError(f'Invalid scene={scene}. Please select one of {scene_names}.')
    
    filename = os.path.join(__CONFIGROOT, f'{scene}.txt')
    with open(filename, 'r') as file:
        lines = file.readlines()
    content = ''.join([line.replace(' =', ':', 1) for line in lines])
    content = StringIO(content)
    return dict(OmegaConf.load(content))
        

class RayDataset(Dataset):
    @staticmethod
    def __embed(x: torch.Tensor, L: int):
        n, d = x.shape
        pi_x = torch.pi * x
        pi_x = pi_x.unsqueeze(-2).expand(n, L, d)
        coef = torch.pow(2, torch.arange(L))
        coef = coef.unsqueeze(-1).expand(L, d)
        coef = coef.unsqueeze(0).expand(n, L, d)

        period_out = torch.stack([
            torch.sin(coef * pi_x), torch.cos(coef * pi_x)
        ], dim=-1).flatten(start_dim=1)
        out = torch.cat([x, period_out], dim=1)
        return out
    
    def __init__(self, dataset, scene, split='train', L_rays=10, L_rgb=4):
        self.dataset = dataset
        self.scene = scene
        self.split = split
        self.L_rays = L_rays
        self.L_rgb = L_rgb
        
        config = _get_config(scene)
        datadir = _get_datadir(dataset, scene)
        dataset_cls = {
            SYNTHETIC: BlenderDataset,
            LLFF_DATA: LlffDataset,
        }[dataset]
        self.__dataset = dataset_cls(data_dir=datadir, split=split, **config)
    
    def __getitem__(self, idx):
        rays, rgbs = self.__dataset[idx]
        x = RayDataset.__embed(rays[0], L=10)
        d = RayDataset.__embed(rays[1], L=4)
        rays = torch.cat([x, d], dim=1)
        return rays, rgbs
    
    def __len__(self):
        return len(self.__dataset)


dataset = RayDataset(LLFF_DATA, 'fern', split='train')
rays, rgbs = dataset[0]
print(rays.shape)
print(rgbs.shape)

dataset = RayDataset(LLFF_DATA, 'fern', split='val')
rays, rgbs = dataset[0]
print(rays.shape)
print(rgbs.shape)

# import matplotlib.pyplot as plt
# plt.imshow(images.view(32, 32, 3))
# plt.savefig('temp/temp.png')

# dataset = RayDataset(SYNTHETIC, 'lego')
# rays, images = dataset[0]
# print(rays.shape)
# print(images.shape)
