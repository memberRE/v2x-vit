import argparse
import os
import time

import easydict
import torch
import open3d as o3d
from torch.utils.data import DataLoader
import tqdm

import v2xvit.hypes_yaml.yaml_utils as yaml_utils
from v2xvit.compress.compression import CompressTools
from v2xvit.compress.models.utils import AverageMeter
from v2xvit.tools import train_utils, infrence_utils
from v2xvit.data_utils.datasets import build_dataset
from v2xvit.visualization import vis_utils
from v2xvit.utils import eval_utils

DEBUG = True

LF_dict = easydict.EasyDict(
    {'hypes_yaml': '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_late_fusion.yaml',
     # 'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/fintuned-dpcc-EF-perfect-bpp4',
     'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/point_pillar_late_fusion_2023_03_05_23_55_23',
     'fusion_method': 'late',
     'save_npy': False,
     'save_vis': False,
     'show_vis': False,
     'show_sequence': False,
     'load_epoch': 11,
     'stage': 'stage1',
     'compress_yaml': None,
     'compress_model': None
     })


EF_dict = easydict.EasyDict(
    {
     # 'hypes_yaml': '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_early_fusion.yaml',
     'hypes_yaml': '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_early_fusion.yaml',
     # 'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/fintuned-dpcc-EF-perfect-bpp4',
     # 'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/early_fusion_no_aug',
     'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/point_pillar_early_fusion_baseline',
     'fusion_method': 'early',
     'save_npy': False,
     'save_vis': False,
     'show_vis': False,
     'show_sequence': False,
     'load_epoch': 22,
     'stage': 'stage1',
     # 'compress_yaml': '/home/JJ_Group/cheny/D-PCC/configs/kitti.yaml',
     # 'compress_model': '/home/JJ_Group/cheny/D-PCC/output/bpp18_no_bug/ckpt/ckpt-best.pth'
     # 'compress_yaml': '/home/JJ_Group/cheny/D-PCC/configs/kitti_bpp2.yaml',
     # 'compress_model': '/home/JJ_Group/cheny/D-PCC/output/bpp2/ckpt/ckpt-best.pth'
     'compress_yaml': None,
     'compress_model': None
     })

MF_dict = easydict.EasyDict(
    {'hypes_yaml': '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/stage3_baseline_compress32_noNoise.yaml',
     'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/baseline_compress32_noNoise',
     'fusion_method': 'intermediate',
     'save_npy': True,
     'save_vis': False,
     'show_vis': False,
     'show_sequence': False,
     'load_epoch': 66,
     'compress_yaml': None,
     'compress_model': None,
     'stage': 'stage1'})

MF_compress0_dict = easydict.EasyDict(
    {'hypes_yaml': '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/stage3_baseline_compress0.yaml',
     'model_dir': '/home/JJ_Group/cheny/v2x-vit/v2xvit/logs/baseline_compress0',
     'fusion_method': 'intermediate',
     'save_npy': True,
     'save_vis': False,
     'show_vis': False,
     'show_sequence': False,
     'load_epoch': 60,
     'compress_yaml': None,
     'compress_model': None,
     'stage': 'stage1'})

preset_dict = {
    'LF': LF_dict,
    'EF': EF_dict,
    'MF': MF_dict,
    'MF_compress0': MF_compress0_dict
}

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--hypes_yaml', type=str, required=False, default=None,
                        help='hypes path')
    parser.add_argument('--model_dir', type=str, required=False, default=None,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=False, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--stage', required=False, type=str,
                        default='stage3')
    parser.add_argument('--load_epoch', required=False, type=int,
                        default=None)
    parser.add_argument('--compress_yaml', type=str, default=None, help='compress yaml file')
    parser.add_argument('--compress_model', default='', help='model path')
    opt = parser.parse_args()
    return opt

def debug_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    # parser.add_argument('--hypes_yaml', type=str, default=None,
    #                     help='hypes path')
    # parser.add_argument('--model_dir', type=str, default=None,
    #                     help='Continued training path')
    parser.add_argument('--overlap', type=str, default=None, help='EF MF LF MF_compress0')
    parser.add_argument('--compress_yaml', type=str, default=None, help='compress yaml file')
    parser.add_argument('--compress_model', default=None, help='model path')
    parser.add_argument('--new_eval', action='store_true',)
    # parser.add_argument('--load_epoch', required=False, type=int,
    #                     default=None)

    # parser.add_argument('--sample_points', type=int, default=None)
    # parser.add_argument('--sample_rates', type=float, default=None)
    # parser.add_argument('--use_strange_sample_mode', action='store_true')
    # parser.add_argument('--spare', action='store_true')
    # parser.add_argument('--mask_first', action='store_true')
    # parser.add_argument('--spare_mode', type=str, default=None)
    # parser.add_argument('--label_mode', type=str, default=None)
    # parser.add_argument('--use_label_mask', action='store_true')
    # parser.add_argument('--compress_ego_car', action='store_true')
    # srun python inference.py --sample_points 4096 --spare --mask_first --spare_mode FPS
    # srun python inference.py --sample_rates 0.3 --spare --mask_first --spare_mode FPS --label_mode foreground_all --use_label_mask
    # self.sample_points = 4096
    # self.sample_rates = -1
    # self.use_strange_sample_mode = False
    # # if True, sample_points = sample_rate*num_points_foreground, only used in W_FPS with use_label_mask=False
    # self.spare = True
    # self.mask_first = True
    # self.spare_mode = 'FPS'
    # self.label_mode = 'bin_CE'
    # self.use_label_mask = False
    # self.compress_ego_car = False
    opt = parser.parse_args()
    return opt

def hyper_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--sample_points', type=int, default=None)
    parser.add_argument('--sample_rates', type=float, default=None)
    parser.add_argument('--use_strange_sample_mode', action='store_true')
    parser.add_argument('--spare', action='store_true')
    parser.add_argument('--mask_first', action='store_true')
    parser.add_argument('--spare_mode', type=str, default=None)
    parser.add_argument('--label_mode', type=str, default=None)
    parser.add_argument('--use_label_mask', action='store_true')
    parser.add_argument('--compress_ego_car', action='store_true')
    opt = parser.parse_args()
    return opt


def update_hypes(opt, expr):
    for k, v in expr.__dict__.items():
        if v is not None:
            opt[k] = v
    return opt


def main():
    print(os.path.abspath('.'))

    num_workers = 8
    use_test = True
    extra_hype = None

    if DEBUG:
        extr = debug_parser()
        if extr.overlap:
            opt = preset_dict[extr.overlap]
        opt = update_hypes(opt, extr)
        # extra_hype = hyper_parser()
        print(opt)
    else:
        opt = test_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print('-------------------------------------------')
    print('load hypes from {}'.format(opt.hypes_yaml))
    print('-------------------------------------------')
    if extra_hype is not None:
        hypes = update_hypes(hypes, extra_hype)
        print(hypes)
    print('-------------------------------------------')
    if use_test:
        hypes['validate_dir'] = '/home/JJ_Group/datasets/V2X/v2xset/test'
        print('use test set')
    else:
        print('use val set')

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    use_compress = True if opt.compress_yaml else False
    bpps = AverageMeter()
    compress_size = AverageMeter()
    if use_compress:
        compress_hypes = yaml_utils.load_yaml(opt.compress_yaml, opt)
        compress_hypes = easydict.EasyDict(compress_hypes)
        # compress_hypes.downsample_rate = [1 / 3, 1 / 3, 1 / 3]
        compress_model = CompressTools(compress_hypes, hypes['preprocess']['cav_lidar_range'], opt.compress_model,  # TODO: check
                                       use_patch=True)
        print('-------------compress model loaded-------------')
        opencood_dataset.set_compress_model(compress_model)
    else:
        compress_hypes = None
        compress_model = None

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model, load_epoch=opt.load_epoch)
    print('Model Loaded , epoch: {}'.format(_))
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_short = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_middle = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    result_stat_long = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    pbar2 = tqdm.tqdm(total=len(data_loader), leave=True)
    for i, batch_data in enumerate(data_loader):
        pbar2.update(1)
        with torch.no_grad():
            if device == 'cuda':
                torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)
            if opt.fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, sizes = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
                compress_size.update(sizes)
            elif opt.fusion_method == 'early':
                bpps.update(batch_data['ego']['bpp_stack'])
                compress_size.update(batch_data['ego']['size_stack'])
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif opt.fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

            # short range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,
                                       left_range=0,
                                       right_range=30)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.7,
                                       left_range=0,
                                       right_range=30)

            # middle range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,
                                       left_range=30,
                                       right_range=50)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.7,
                                       left_range=30,
                                       right_range=50)

            # right range
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,
                                       left_range=50,
                                       right_range=100)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.7,
                                       left_range=50,
                                       right_range=100)

            if opt.save_npy:
                if opt.compress_model:
                    dir_name = opt.compress_model.split('/')[-3]
                else:
                    dir_name = 'npy'
                npy_save_path = os.path.join(opt.model_dir, dir_name)
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                infrence_utils.save_prediction_gt(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  i,
                                                  npy_save_path)

            if opt.show_vis or opt.save_vis:
                vis_save_path = ''
                if opt.save_vis:
                    vis_save_path = os.path.join(opt.model_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                opencood_dataset.visualize_result(pred_box_tensor,
                                                  gt_box_tensor,
                                                  batch_data['ego'][
                                                      'origin_lidar'][0],
                                                  opt.show_vis,
                                                  vis_save_path,
                                                  dataset=opencood_dataset)

            if opt.show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        vis_pcd,
                        mode='constant'
                    )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_pred,
                                                 pred_o3d_box,
                                                 update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                 vis_aabbs_gt,
                                                 gt_o3d_box,
                                                 update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat,
                                  opt.model_dir,
                                  bpps=bpps.get_avg(),
                                  compress_size=compress_size.get_avg(),
                                  resort=opt.new_eval)
    eval_utils.eval_final_results(result_stat_short,
                                  opt.model_dir,
                                  range="short", resort=opt.new_eval)
    eval_utils.eval_final_results(result_stat_middle,
                                  opt.model_dir,
                                  range="middle", resort=opt.new_eval)
    eval_utils.eval_final_results(result_stat_long,
                                  opt.model_dir,
                                  range="long", resort=opt.new_eval)
    repr_bpp = 'bpp: {}, compress_size: {}'.format(bpps.get_avg(), compress_size.get_avg())
    res_name = 'result_' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '.txt'
    with open(os.path.join(saved_path, res_name), 'a+') as f:
        msg = 'Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} | comm_rate: {:.06f} | comm_bits: {:.04f} | comm_bits_ori: {:.04f}\n'.format(
            -1, ap_30, ap_50, ap_70, 0, 0, -1)
        f.write(msg)
        print(msg)
    print(repr_bpp)
    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
