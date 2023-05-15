import os, sys
if __name__ == '__main__':
    sys.path.append(os.getcwd())

from abc import ABC, abstractmethod
from typing import Iterable, Union, OrderedDict
from models.blending.serialization import Serializer

import torch
from torch import nn


class BlendingPoolBase(ABC):
    def __init__(self, experts: Iterable[Union[nn.Module, OrderedDict]]):
        if len(experts) < 2:
            raise RuntimeError('More than one experts are required.')
        
        self._serializer = Serializer.from_hint(experts[0])
        self._experts = self._serializer(experts)
        
    @property
    def num_experts(self):
        return self._experts.size(0)
    
    @property
    def expert_dim(self):
        return self._experts.size(1)
    
    @property
    def serializer(self):
        return self._serializer
        
    @abstractmethod
    def blend(self, weights: torch.Tensor, deserialize: bool=False):
        pass
        
    @abstractmethod
    def __len__(self):
        pass
    
    def __str__(self):
        return f'{str(type(self).__name__)}(len={len(self)})'
    
    
class ElementBlendingPool(BlendingPoolBase):
    def __init__(self, experts: Iterable[Union[nn.Module, OrderedDict]]):
        super(ElementBlendingPool, self).__init__(experts)
        self.__len = self._experts.size(1)

    def blend(self, weights: torch.Tensor, deserialize: bool=False):
        if self._experts.shape != weights.shape:
            raise RuntimeError('The shape of the weight must be (num_experts, flattened_expert_size).')
        
        weighted_experts = self._experts * weights
        new_expert = weighted_experts.sum(dim=0, keepdim=True)  # (1, dim)
        if deserialize:
            new_expert = next(self._serializer.deserialize(new_expert))
        return new_expert
        
    def __len__(self):
        return self.__len
    
    
class TensorBlendingPool(BlendingPoolBase):
    def __init__(self, experts: Iterable[Union[nn.Module, OrderedDict]]):
        super(TensorBlendingPool, self).__init__(experts)
        self.__len = len(experts[0].state_dict() if isinstance(experts[0], nn.Module) else experts[0])
        
    def blend(self, weights: torch.Tensor, deserialize: bool=False):
        if self._experts.size(0) != weights.size(0):
            raise RuntimeError('The number of experts and the number of weights must be equal.')
        if self.__len != weights.size(1):
            raise RuntimeError('The number of parameter tensors and the dimension of a weight must be equal.')
        
        new_expert = []
        for offset, size, weight in zip(self._serializer.offsets, self._serializer.sizes, weights.T):
            params = self._experts[:, offset:offset+size]
            new_params = (params * weight[:, None]).sum(dim=0, keepdim=True)
            new_expert.append(new_params)
        new_expert = torch.cat(new_expert, dim=1)
        
        if deserialize:
            new_expert = next(self._serializer.deserialize(new_expert))
        return new_expert
        
    def __len__(self):
        return self.__len


pool_map = {
    'element': ElementBlendingPool,
    'tensor': TensorBlendingPool,
}
pool_names = list(pool_map.keys())


if __name__ == '__main__':
    device = torch.device('cuda:0')
    
    sample_x = torch.randn(4, 10).to(device)
    generate_expert = lambda: nn.Sequential(
        nn.Linear(sample_x.size(1), 20), nn.ReLU(inplace=True),
        nn.Linear(20, 5), nn.ReLU(inplace=True),
        nn.Linear(5, 10), nn.Sigmoid(),
    ).to(device)
    expert_scheme = {key: val.shape for key, val in generate_expert().state_dict().items()}
    
    n_experts = 5
    experts = [generate_expert().state_dict() for _ in range(n_experts)]

    net = generate_expert()
    for name in pool_names:
        print(name)
        pool = pool_map[name](experts)
        weights = nn.Parameter(torch.ones(pool.num_experts, len(pool), requires_grad=True).to(device))
        print('weight:', weights.shape)
        expert = pool.blend(weights)
        print('expert:', expert.shape)
        print()
