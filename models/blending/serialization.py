from typing import Iterable, Union, OrderedDict

import torch
from torch import nn


class Serializer:
    @staticmethod
    def from_hint(hint: Union[nn.Module, OrderedDict]):
        if isinstance(hint, nn.Module):
            hint = hint.state_dict()
        keys, shapes = [], []
        for key, val in hint.items():
            keys.append(key)
            shapes.append(val.shape)
        return Serializer(keys, shapes)
    
    def __init__(self, keys, shapes):
        if len(keys) != len(shapes):
            raise RuntimeError('Expected same number of keys and shapes.')
        
        self.__keys = tuple(keys)
        self.__shapes = tuple(tuple(sh) for sh in shapes)
        self.__sizes = tuple(torch.tensor(sh).prod().int().item() for sh in self.__shapes)
        self.__offsets = tuple(torch.tensor([0] + list(self.__sizes)).cumsum(dim=0)[:-1].tolist())
        
    def serialize(self, state_dicts: Iterable[OrderedDict], acquire_grad: bool=False):
        if acquire_grad:
            get_fn = lambda x: x.grad
        else:
            get_fn = lambda x: x
        
        return torch.stack([
            torch.cat([get_fn(param).view(-1) for param in state_dict.values()], dim=0)
            for state_dict in state_dicts], dim=0)
    
    def deserialize(self, params: torch.Tensor):
        for param in params:
            yield OrderedDict({
                key: param[offset:offset+size].reshape(shape)
                for key, offset, size, shape 
                in zip(self.__keys, self.__offsets, self.__sizes, self.__shapes)
            })
            
    @property
    def keys(self):
        return self.__keys
    
    @property
    def offsets(self):
        return self.__offsets
        
    @property
    def sizes(self):
        return self.__sizes
    
    @property
    def shapes(self):
        return self.__shapes
        
    @property
    def metadata(self):
        return dict(zip(self.__keys, zip(self.__offsets, self.__sizes, self.__shapes)))
    
    def __call__(self, state_dicts: Iterable[OrderedDict]):
        return self.serialize(state_dicts)
    
    def __str__(self):
        lines = ['Serializer(', '  # key: (offset, size, shape) #'] + \
            [f'    {key}: {val},' for key, val in self.metadata.items()] + [')']
        return '\n'.join(lines)

    def __repr__(self):
        return str(self)
        

if __name__ == '__main__':
    make_expert = lambda: nn.Linear(1, 2).state_dict()

    hint = make_expert()
    serializer = Serializer.from_hint(hint)
    print(serializer)
    print()

    experts = [make_expert() for _ in range(3)]
    print(experts)
    print()

    params = serializer(experts)
    print(params)
    print()

    recovered = list(serializer.deserialize(params))
    print(recovered)
