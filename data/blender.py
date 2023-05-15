import os
import json
import numpy as np
import torch
import imageio
import cv2
import time
from torch.utils.data import Dataset

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

class BlenderDataset(Dataset):
    def __init__(self, data_dir, half_res=False, testskip=1, white_bkgd=False, N_rand=None, split='train', **kwargs):
        self.data_dir = data_dir
        self.half_res = half_res
        self.testskip = testskip
        self.white_bkgd = white_bkgd
        self.N_rand = N_rand
        self.near = 2.
        self.far = 6.
        images, self.poses, self.render_poses, self.hwf, self.self_split_indices = \
            self.load_blender_data(basedir=data_dir, half_res=half_res, testskip=testskip)
        
        if white_bkgd:
            self.images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            self.images = images[...,:3]
        
        i_train, i_val, i_test = self.self_split_indices
        self.i_train = {
            'train': i_train,
            'val': i_val,
            'test': i_test,
        }[split]

        
    def __len__(self):
        return 1
    
    def __getitem__(self, _):
        rays, target_s = self.generate_rays()
        rays = torch.from_numpy(rays).float()
        target_s = torch.from_numpy(target_s).float()
        return rays, target_s
    
    def load_blender_data(self, basedir, half_res=False, testskip=1):
        splits = ['train', 'val', 'test']
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            if s=='train' or testskip==0:
                skip = 1
            else:
                skip = testskip
                
            for frame in meta['frames'][::skip]:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            poses = np.array(poses).astype(np.float32)
            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
        
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        
        if half_res:
            H = H//2
            W = W//2
            focal = focal/2.

            imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
            # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

            
        return imgs, poses, render_poses, [H, W, focal], i_split

    def generate_rays(self):
        
        H, W, focal = self.hwf
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        np.random.seed(int(time.time() * 1e6) % (2**32))
        img_i = np.random.choice(self.i_train)
        
        # Randomly choose one image index from the training set
        target = self.images[img_i]
        pose = self.poses[img_i, :3, :4]

        rays_o, rays_d = self.get_rays_np(H, W, K, pose)  # (H, W, 3), (H, W, 3)

        if self.N_rand is not None:
            dH = int(H//2 * 0.5)
            dW = int(W//2 * 0.5)
            coords = np.stack(
                np.meshgrid(
                    np.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    np.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1)
            # coords = np.stack(np.meshgrid(np.linspace(0, H - 1, H), np.linspace(0, W - 1, W)), -1)  # (H, W, 2)
            coords = np.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].astype(int)  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            rays = np.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
            #  rays = rays.transpose(1, 0, 2)  # (N_rand, 2, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]
        return rays, target_s
    
    def get_rays_np(self, H, W, K, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        return rays_o, rays_d