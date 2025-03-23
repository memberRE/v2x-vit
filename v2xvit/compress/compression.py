import time

import numpy as np
import os
import torch
from easydict import EasyDict

from v2xvit.compress.models.autoencoder import AutoEncoder
from v2xvit.compress.metrics.PSNR import get_psnr
from v2xvit.compress.metrics.density import get_density_metric
from v2xvit.compress.metrics.F1Score import get_f1_score
from v2xvit.compress.models.utils import AverageMeter
from v2xvit.hypes_yaml import yaml_utils
from v2xvit.utils import pcd_utils
from v2xvit.utils.pcd_utils import mask_points_by_range


def normalize_pcd(xyzs):
    '''
     normalize xyzs to [0,1], keep ratio unchanged
    '''
    shift = np.mean(xyzs, axis=0)  # 1 x 3
    xyzs -= shift
    max_coord, min_coord = np.max(xyzs), np.min(xyzs)
    xyzs = xyzs - min_coord
    xyzs = xyzs / (max_coord - min_coord)
    meta_data = {}
    meta_data['shift'] = shift
    meta_data['max_coord'] = max_coord
    meta_data['min_coord'] = min_coord
    return xyzs, meta_data


def divide_cube(xyzs, attribute, map_size=100, cube_size=12):
    '''
    xyzs: N x 3
    resolution: 100 x 100 x 100
    cube_size: 10 x 10 x 10
    min_num and max_num points in each cube, if small than min_num or larger than max_num then discard it
    return label that indicates each points' cube_idx
    '''

    output_points = {}
    # points = np.dot(points, get_rotate_matrix())
    xyzs, meta_data = normalize_pcd(xyzs)

    map_xyzs = xyzs * (map_size)
    xyzs = np.floor(map_xyzs).astype('float32')

    cubes = {}
    for idx, point_idx in enumerate(xyzs):
        # the cube_idx is a 3-dim tuple
        tuple_cube_idx = tuple((point_idx // cube_size).astype(int))
        if not tuple_cube_idx in cubes.keys():
            cubes[tuple_cube_idx] = []
        cubes[tuple_cube_idx].append(idx)

    for tuple_cube_idx, point_idx in cubes.items():
        dim_cube_num = np.ceil(map_size / cube_size).astype(int)
        # indicate which cube a point belongs to
        cube_idx = tuple_cube_idx[0] * dim_cube_num * dim_cube_num + \
                   tuple_cube_idx[1] * dim_cube_num + tuple_cube_idx[2]

        output_points[cube_idx] = np.concatenate([map_xyzs[point_idx, :], attribute[point_idx, :]], axis=-1)

    # label indicates each points' cube index
    # label may have some value less than 0, which should be ignored
    return output_points, meta_data


class CompressTools:
    def __init__(self, cfg, range, model_path=None, dataset=None, train=False, use_patch=False, mapsize=100):
        self.cfg = cfg
        self.dataset = dataset
        self.train = train
        self.model = self.load_model(cfg, model_path)
        self.range = np.array(range[3:]) - np.array(range[:3])  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.range = torch.from_numpy(self.range).float().cuda()
        self.bppavg = AverageMeter()
        self.compress_size = AverageMeter()  # bits
        self.use_patch = use_patch
        self.mapsize = mapsize
        self.cube_size = cfg.train_cube_size
        self.dim_cube_num = np.ceil(mapsize / self.cube_size).astype(int)

    def load_model(self, args, model_path):
        # load model
        model = AutoEncoder(args).cuda()
        params = torch.load(model_path)
        for i in params.keys():
            if i.startswith('feats_eblock') or i.startswith('xyzs_eblock'):
                if i.endswith('_offset') or i.endswith('_quantized_cdf') or i.endswith('_cdf_length'):
                    params[i] = torch.tensor([], dtype=params[i].dtype, device=params[i].device)
        model.load_state_dict(torch.load(model_path))
        # update entropy bottleneck
        model.feats_eblock.update(force=True)
        if args.quantize_latent_xyzs == True:
            model.xyzs_eblock.update(force=True)
        model.eval()

        return model

    def fakeCompress(self, input, normals):
        with torch.no_grad():
            # normalize xyzs
            # input, center = self.normalize(xyzs)

            gt_normals = normals
            input = input.permute(0, 2, 1).contiguous()
            # concat normals
            input = torch.cat((input, gt_normals.permute(0, 2, 1).contiguous()), dim=1)
            xyzs = input[:, :3, :].contiguous()
            gt_patches = xyzs
            feats = input
            points_num = xyzs.shape[0] * xyzs.shape[2]
            latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
            actual_bpp = self.compress(xyzs.float(), feats.float())

            pred_patches, upsampled_feats, decode_time \
                = self.decompress(latent_xyzs_str, xyzs_size, latent_feats_str, feats_size)

            # pred_normals = torch.tanh(upsampled_feats).permute(0, 2, 1).contiguous()
            # pred_normals = torch.mean(pred_normals, dim=2, keepdim=True)
            # TODO: use sigmoid instead of tanh
            pred_normals = torch.sigmoid(upsampled_feats).permute(0, 2, 1).contiguous()

            pred_normals = torch.clamp(pred_normals, 0, 1)

            # gt_patches = gt_patches.permute(0, 2, 1).contiguous()
            pred_patches = pred_patches.permute(0, 2, 1).contiguous()
            # pred_patches = self.denormalize(pred_patches, center)
            # output = torch.cat((pred_patches, pred_normals), dim=2)
            self.bppavg.update(actual_bpp)
            # self.compress_size.update(actual_bpp * points_num)
            return pred_patches, pred_normals, actual_bpp, actual_bpp * points_num

    def compress(self, xyzs, feats):
        # input: (b, c, n)

        encode_start = time.time()
        # raise dimension
        feats = self.model.pre_conv(feats)

        # encoder forward
        gt_xyzs, gt_dnums, gt_mdis, latent_xyzs, latent_feats = self.model.encoder(xyzs, feats)
        # decompress size
        feats_size = latent_feats.size()[2:]

        # compress latent feats
        latent_feats_str = self.model.feats_eblock.compress(latent_feats)

        # compress latent xyzs
        if self.cfg.quantize_latent_xyzs == True:
            analyzed_latent_xyzs = self.model.latent_xyzs_analysis(latent_xyzs)
            # decompress size
            xyzs_size = analyzed_latent_xyzs.size()[2:]
            latent_xyzs_str = self.model.xyzs_eblock.compress(analyzed_latent_xyzs)
        else:
            # half float representation
            latent_xyzs_str = latent_xyzs.half()
            xyzs_size = None

        encode_time = time.time() - encode_start

        # bpp calculation
        points_num = xyzs.shape[0] * xyzs.shape[2]
        feats_bpp = (sum(len(s) for s in latent_feats_str) * 8.0) / points_num
        if self.cfg.quantize_latent_xyzs == True:
            xyzs_bpp = (sum(len(s) for s in latent_xyzs_str) * 8.0) / points_num
        else:
            xyzs_bpp = (latent_xyzs.shape[0] * latent_xyzs.shape[2] * 16 * 3) / points_num
        actual_bpp = feats_bpp + xyzs_bpp

        return latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, actual_bpp

    def decompress(self, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size):
        decode_start = time.time()
        # decompress latent xyzs
        if self.cfg.quantize_latent_xyzs == True:
            analyzed_latent_xyzs_hat = self.model.xyzs_eblock.decompress(latent_xyzs_str, xyzs_size)
            latent_xyzs_hat = self.model.latent_xyzs_synthesis(analyzed_latent_xyzs_hat)
        else:
            latent_xyzs_hat = latent_xyzs_str

        # decompress latent feats
        latent_feats_hat = self.model.feats_eblock.decompress(latent_feats_str, feats_size)

        # decoder forward
        pred_xyzs, pred_unums, pred_mdis, upsampled_feats = self.model.decoder(latent_xyzs_hat, latent_feats_hat)

        decode_time = time.time() - decode_start

        return pred_xyzs[-1], upsampled_feats, decode_time

    def __call__(self, lidar):
        lidar_xyz = lidar[:, :3]
        lidar_normals = lidar[:, 3:]
        cubes, meta_data = divide_cube(lidar_xyz, attribute=lidar_normals, cube_size=self.cube_size)

        points = []
        sizes = 0
        bpps = AverageMeter()

        for patch_idx in cubes.keys():
            if cubes[patch_idx].shape[0] < 100:
                points.append(cubes[patch_idx])
                continue
            cube_x = patch_idx // self.dim_cube_num ** 2
            cube_y = (patch_idx - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
            cube_z = patch_idx % self.dim_cube_num
            center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
            xyzs = cubes[patch_idx][:, :3]  # absolute coordinate(quantized)
            normals = cubes[patch_idx][:, 3:]
            xyzs = 2 * (xyzs - center) / self.cube_size
            xyzs = torch.tensor(xyzs).float().cuda()
            normals = torch.tensor(normals).float().cuda()
            pred_patches, pred_normals, bpp, compress_size_patch = self.fakeCompress(xyzs[None, ...],
                                                                                     normals[None, ...])
            sizes += compress_size_patch
            bpps.update(bpp)

            pred_patches = pred_patches * self.cube_size / 2 + torch.tensor(center).float().to(pred_patches.device)
            pred_patches = pred_patches / 100
            pred_normals = pred_normals.cpu()
            pred_patches = pred_patches.cpu()
            pred_patches = pred_patches * (meta_data['max_coord'] - meta_data['min_coord']) + meta_data['min_coord'] + \
                           meta_data['shift']
            pred_patches = pred_patches.squeeze(0).detach()
            pred_normals = pred_normals.squeeze(0).detach()
            out = torch.cat((pred_patches, pred_normals), dim=1).cpu().numpy()
            points.append(out)

        pred_pcd = np.concatenate(points, axis=0)
        self.compress_size.update(sizes)
        return pred_pcd, bpps.get_avg(), sizes

    def __repr__(self):
        return 'avg_bpp: {:.4f}, avg_compress_size: {:.4f} bits'.format(self.bppavg.avg, self.compress_size.avg)

# def test_with_FPS():
#

if __name__ == '__main__':
    model_path = '/home/JJ_Group/cheny/D-PCC/output/2023-01-09T19:04:01.790133/ckpt/ckpt-epoch-20.pth'
    cfg = '/home/JJ_Group/cheny/D-PCC/configs/kitti.yaml'
    cfg = yaml_utils.load_yaml(cfg)
    cfg = EasyDict(cfg)
    cfg.downsample_rate = [1 / 3, 1 / 3, 1 / 3]
    range2 = np.array([-140.8, -40, -3, 140.8, 40, 1])
    compressor = CompressTools(cfg, range2, model_path, use_patch=True)

    lidar = pcd_utils.pcd_to_np('/home/JJ_Group/cheny/000110.pcd')
    lidar = mask_points_by_range(lidar, range2)
    lidar = compressor(lidar)
    np.save(os.path.join('./', 'pcd.npy'), lidar)
