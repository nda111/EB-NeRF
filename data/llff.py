import os
import numpy as np
import torch
import imageio
import time
from torch.utils.data import Dataset

class LlffDataset(Dataset):
    def __init__(self, data_dir, factor, recenter=True, bd_factor=.75, spherify=False,
                 N_rand = None, path_zfloat=False, llffhold = 8, no_ndc = True, split='train', **kwargs):
        
        self.N_rand = N_rand
        self.images, poses, bds, render_poses, i_test = self.load_llff_data(data_dir, factor,
                                                                  recenter=recenter, bd_factor=bd_factor,
                                                                  spherify=spherify)
        self.hwf = poses[0,:3,-1]
        self.poses = poses[:,:3,:4]
        
        if not isinstance(i_test, list):
            i_test = [i_test]

        if llffhold > 0:
            print('Auto LLFF holdout,', llffhold)
            i_test = np.arange(self.images.shape[0])[::llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(self.images.shape[0])) if
                        (i not in i_test and i not in i_val)])
        if no_ndc:
            self.near = np.ndarray.min(bds) * .9
            self.far = np.ndarray.max(bds) * 1.
        else:
            self.near = 0.
            self.far = 1.
        
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
    
    def load_llff_data(self, basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
        
        poses, bds, imgs = self._load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
        print('Loaded', basedir, bds.min(), bds.max())
        
        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        images = imgs
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        
        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc
        
        if recenter:
            poses = self.recenter_poses(poses)
            
        if spherify:
            poses, render_poses, bds = self.spherify_poses(poses, bds)
        else:            
            c2w = self.poses_avg(poses)
            print('recentered', c2w.shape)
            print(c2w[:3,:4])

            ## Get spiral
            # Get average pose
            up = self.normalize(poses[:, :3, 1].sum(0))

            # Find a reasonable "focus depth" for this dataset
            close_depth, inf_depth = bds.min()*.9, bds.max()*5.
            dt = .75
            mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
            focal = mean_dz

            # Get radii for spiral path
            shrink_factor = .8
            zdelta = close_depth * .2
            tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = c2w
            N_views = 120
            N_rots = 2
            if path_zflat:
    #             zloc = np.percentile(tt, 10, 0)[2]
                zloc = -close_depth * .1
                c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
                rads[2] = 0.
                N_rots = 1
                N_views/=2

            # Generate poses for spiral path
            render_poses = self.render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
            
            
        render_poses = np.array(render_poses).astype(np.float32)

        c2w = self.poses_avg(poses)
        print('Data:')
        print(poses.shape, images.shape, bds.shape)
        
        dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
        i_test = np.argmin(dists)
        print('HOLDOUT view is', i_test)
        
        images = images.astype(np.float32)
        poses = poses.astype(np.float32)

        return images, poses, bds, render_poses, i_test
    
    def generate_rays(self):
        H, W, focal = self.hwf
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        # np.random.seed(int(time.time() * 1e6) % (2**32))
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
            
            near = self.near * np.ones_like(rays_d[..., :1])
            far = self.far * np.ones_like(rays_d[..., :1])
            rays = np.concatenate([rays_o, rays_d, near, far], axis=-1)

            rays = np.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
            #  rays = rays.transpose(1, 0, 2)  # (N_rand, 2, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]
        return rays, target_s
    
    def _load_data(self, basedir, factor=None, width=None, height=None, load_imgs=True):
        poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -2:].transpose([1,0])
        
        img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
        sh = imageio.imread(img0).shape
        
        sfx = ''
        if factor is not None:
            sfx = '_{}'.format(factor)
            self._minify(basedir, factors=[factor])
            factor = factor
        elif height is not None:
            factor = sh[0] / float(height)
            width = int(sh[1] / factor)
            self._minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        elif width is not None:
            factor = sh[1] / float(width)
            height = int(sh[0] / factor)
            self._minify(basedir, resolutions=[[height, width]])
            sfx = '_{}x{}'.format(width, height)
        else:
            factor = 1
        
        imgdir = os.path.join(basedir, 'images' + sfx)
        if not os.path.exists(imgdir):
            print( imgdir, 'does not exist, returning' )
            return
        
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        if poses.shape[-1] != len(imgfiles):
            print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
            return
        
        sh = imageio.imread(imgfiles[0]).shape
        poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1./factor
        
        if not load_imgs:
            return poses, bds
        
        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
            
        imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
        imgs = np.stack(imgs, -1)  
        
        print('Loaded image data', imgs.shape, poses[:,-1,0])
        return poses, bds, imgs
    
    def get_rays_np(self, H, W, K, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        return rays_o, rays_d
        ##################################################
    
    def _minify(self, basedir, factors=[], resolutions=[]):
        needtoload = False
        for r in factors:
            imgdir = os.path.join(basedir, 'images_{}'.format(r))
            if not os.path.exists(imgdir):
                needtoload = True
        for r in resolutions:
            imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
            if not os.path.exists(imgdir):
                needtoload = True
        if not needtoload:
            return
        
        from subprocess import check_output
        
        imgdir = os.path.join(basedir, 'images')
        imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
        imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
        imgdir_orig = imgdir
        
        wd = os.getcwd()

        for r in factors + resolutions:
            if isinstance(r, int):
                name = 'images_{}'.format(r)
                resizearg = '{}%'.format(100./r)
            else:
                name = 'images_{}x{}'.format(r[1], r[0])
                resizearg = '{}x{}'.format(r[1], r[0])
            imgdir = os.path.join(basedir, name)
            if os.path.exists(imgdir):
                continue
                
            print('Minifying', r, basedir)
            
            os.makedirs(imgdir)
            check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
            
            ext = imgs[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
            print(args)
            os.chdir(imgdir)
            check_output(args, shell=True)
            os.chdir(wd)
            
            if ext != 'png':
                check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
                print('Removed duplicates')
            print('Done')
    
    def recenter_poses(self, poses):
    
        poses_ = poses+0
        bottom = np.reshape([0,0,0,1.], [1,4])
        c2w = self.poses_avg(poses)
        c2w = np.concatenate([c2w[:3,:4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
        poses = np.concatenate([poses[:,:3,:4], bottom], -2)

        poses = np.linalg.inv(c2w) @ poses
        poses_[:,:3,:4] = poses[:,:3,:4]
        poses = poses_
        return poses
    
    def spherify_poses(self, poses, bds):
        
        p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
        
        rays_d = poses[:,:3,2:3]
        rays_o = poses[:,:3,3:4]

        def min_line_dist(rays_o, rays_d):
            A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
            b_i = -A_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        
        center = pt_mindist
        up = (poses[:,:3,3] - center).mean(0)

        vec0 = self.normalize(up)
        vec1 = self.normalize(np.cross([.1,.2,.3], vec0))
        vec2 = self.normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)

        poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
        
        sc = 1./rad
        poses_reset[:,:3,3] *= sc
        bds *= sc
        rad *= sc
        
        centroid = np.mean(poses_reset[:,:3,3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad**2-zh**2)
        new_poses = []
        
        for th in np.linspace(0., 2.*np.pi , 120):

            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])

            vec2 = self.normalize(camorigin)
            vec0 = self.normalize(np.cross(vec2, up))
            vec1 = self.normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)

            new_poses.append(p) 

        new_poses = np.stack(new_poses, 0)
        
        new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
        poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
        
        return poses_reset, new_poses, bds
    
    def render_path_spiral(self, c2w, up, rads, focal, zdelta, zrate, rots, N):
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w[:,4:5]
        
        for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
            c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
            z = self.normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
            render_poses.append(np.concatenate([self.viewmatrix(z, up, c), hwf], 1))
        return render_poses
    
    
    def poses_avg(self, poses):
        
        hwf = poses[0, :3, -1:]

        center = poses[:, :3, 3].mean(0)
        vec2 = self.normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self.viewmatrix(vec2, up, center), hwf], 1)
        
        return c2w
    
    def normalize(self, x):
        return x / np.linalg.norm(x)
    
    def viewmatrix(self, z, up, pos):
        vec2 = self.normalize(z)
        vec1_avg = up
        vec0 = self.normalize(np.cross(vec1_avg, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m
