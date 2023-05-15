from typing import Literal
import torch
import pytorch_msssim
import lpips as lpipslib

__lpips = {
    'alex': lpipslib.LPIPS(net='alex', verbose=False),
    'vgg': lpipslib.LPIPS(net='vgg', verbose=False),
    'squeeze': lpipslib.LPIPS(net='squeeze', verbose=False),
}


def mean_squared_error(img1: torch.Tensor, img2: torch.Tensor, size_average: bool=True):
    mse = torch.nn.functional.mse_loss(img1, img2, reduction='none')
    mse = torch.mean(mse, dim=tuple(range(1, mse.ndim)))
    if size_average:
        mse = torch.mean(mse)
    return mse

def psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: int=1, size_average: bool=True):
    mse = mean_squared_error(img1, img2, size_average=size_average)
    return 10.0 * torch.log10(data_range / mse)

def ssim(img1: torch.Tensor, img2: torch.Tensor,
         data_range: int=1, size_average: bool=True,
         window_size: int = 11, sigma: float = 1.5, 
         k1: float = 0.01, k2: float = 0.03):
    return pytorch_msssim.ssim(img1, img2, data_range=data_range, size_average=size_average, 
                               win_size=window_size, win_sigma=sigma, K=(k1, k2))

def lpips(img1: torch.Tensor, img2: torch.Tensor, net=Literal['alex', 'vgg'], data_range: int=1, size_average: bool=True):
    value = __lpips[net](img1 / data_range, img2 / data_range).flatten()
    if size_average:
        return value.mean()
    else:
        return value


def __test():
    batch_size = 2
    height, width = 224, 224
    
    img1 = torch.rand(batch_size, 3, height, width, requires_grad=True)
    img2 = torch.rand(batch_size, 3, height, width, requires_grad=True)

    psnr_value = psnr(img1, img2, size_average=False)
    ssim_value = ssim(img1, img2, size_average=False)
    lpips_alex_value = lpips(img1, img2, net='alex', size_average=False)
    lpips_vgg_value = lpips(img1, img2, net='vgg', size_average=False)
    lpips_squeeze_value = lpips(img1, img2, net='squeeze', size_average=False)
    
    print(f'{psnr_value=}')
    print(f'{ssim_value=}')
    print(f'{lpips_alex_value=}')
    print(f'{lpips_vgg_value=}')
    print(f'{lpips_squeeze_value=}')
    
    scores = torch.stack([psnr_value, ssim_value, 
                          lpips_alex_value, lpips_vgg_value, 
                          lpips_squeeze_value], dim=0)
    scores.mean(dim=1).sum().backward()
    print('Backward OK')


if __name__ == '__main__':
    __test()
