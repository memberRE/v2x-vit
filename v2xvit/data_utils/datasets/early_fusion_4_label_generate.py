"""
Dataset class for early fusion
"""
import math
import os
import warnings
from collections import OrderedDict

import numpy as np
import torch

import v2xvit
from v2xvit.hypes_yaml import yaml_utils
from v2xvit.models.pointnet_util import spare_point_cloud
from v2xvit.utils import box_utils
from v2xvit.data_utils.post_processor import build_postprocessor
from v2xvit.data_utils.datasets import basedataset
from v2xvit.data_utils.pre_processor import build_preprocessor
from v2xvit.hypes_yaml.yaml_utils import load_yaml
from v2xvit.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum, mask_point_by_label
from v2xvit.utils.transformation_utils import x1_to_x2

DEBUG = True


class EarlyFusion4LabelGenerate(basedataset.BaseDataset):
    def __init__(self, params, visualize, train=True, compress=None):
        super(EarlyFusion4LabelGenerate, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)
        self.spare = params.get('spare', False)
        self.mask_first = params.get('mask_first', True)
        self.sample_points = params.get('sample_points', -1)
        self.compress_model = compress
        self.sample_rates = params.get('sample_rates', 0)
        self.label_mode = params.get('label_mode', 'foreground_all')
        self.use_label_mask = params.get('use_label_mask', False)
        assert self.label_mode in ['bin_MSE', 'bin_cross_entropy', 'multi_MSE', 'multi_cross_entropy', 'foreground_all']
        # self.recon_mode = params.get('recon_mode', 'perBlock')
        # assert self.recon_mode in ['perBlock', 'allin']

        if DEBUG:
            self.sample_points = -1
            self.sample_rates = 0
            self.spare = True
            self.mask_first = True
            # self.label_mode = 'foreground_all'
            self.label_mode = 'foreground_all'
            self.use_label_mask = True
            print('DEBUG mode, sample points: {}, sample rate: {}, mask_first: {}, label_mode: {}, use_label_mask: {}'
                  .format(self.sample_points, self.sample_rates, self.mask_first, self.label_mode, self.use_label_mask))

    def set_compress_model(self, compress_model):
        self.compress_model = compress_model
        print('using D-PCC compression')

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx, cur_ego_pose_flag=True)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        projected_lidar_stack = []
        object_stack = []
        object_id_stack = []
        bpp_stack = []
        size_stack = []

        spatial_correction_matrix = []

        cav_id = ego_id
        selected_cav_base = base_data_dict[cav_id]

        selected_cav_processed = self.get_item_single_car(
            selected_cav_base,
            ego_lidar_pose,
            base_data_dict[ego_id])
        # all these lidar and object coordinates are projected to ego
        # already.
        lidar = selected_cav_processed['projected_lidar']

        projected_lidar_stack.append(lidar)
        object_stack.append(selected_cav_processed['object_bbx_center'])
        object_id_stack += selected_cav_processed['object_ids']
        spatial_correction_matrix.append(
            selected_cav_base['params']['spatial_correction_matrix'])

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in object_id_stack] # no need to unique
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # convert list to numpy array, (N, 4)
        projected_lidar_stack = np.vstack(projected_lidar_stack)  # about 116875 points

        # data augmentation
        # projected_lidar_stack, object_bbx_center, mask = \
        #     self.augment(projected_lidar_stack, object_bbx_center, mask)  # no need to augment

        # we do lidar filtering in the stacked lidar
        # about 86172 points
        projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                     self.params['preprocess'][
                                                         'cav_lidar_range'])
        # augmentation may remove some of the bbx out of range
        object_bbx_center_valid = object_bbx_center[mask == 1]
        # object_bbx_center_valid = \
        #     box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
        #                                              self.params['preprocess'][
        #                                                  'cav_lidar_range'],
        #                                              self.params[
        #                                                  'postprocess'][
        #                                                  'order']
        #                                              ) # no need to mask
        mask[object_bbx_center_valid.shape[0]:] = 0
        unique_indices = unique_indices[:object_bbx_center_valid.shape[0]]
        object_bbx_center[:object_bbx_center_valid.shape[0]] = \
            object_bbx_center_valid
        object_bbx_center[object_bbx_center_valid.shape[0]:] = 0

        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(projected_lidar_stack)  # about 3000 voxels

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=mask)

        spatial_correction_matrix = np.stack([spatial_correction_matrix[0]])

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'anchor_box': anchor_box,
             'processed_lidar': lidar_dict,
             'spatial_correction_matrix': spatial_correction_matrix,
             'velocity': [0],
             'time_delay': [0],
             'infra': [0],
             'label_dict': label_dict,
             'bpp_stack': bpp_stack,
             'size_stack': size_stack,
             'path': selected_cav_base['path']})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                                                   projected_lidar_stack})

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose, ego_cav_base = None):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix
        transformation_matrix = selected_cav_base['params'][
            'transformation_matrix']

        point_label = self.generate_label_complement(selected_cav_base, ego_cav_base)  # (N)
        point_label = point_label.reshape((-1, 1))  # N x 1

        # retrieve objects under ego coordinates
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = np.hstack((lidar_np, point_label))  # N x 5
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        lidar_np[:, :3] = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        lidar_label = lidar_np[:, -1]
        lidar_np = lidar_np[:, :-1]
        selected_cav_processed.update(
            {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
             'object_ids': object_ids,
             'projected_lidar': lidar_np,
             'lidar_label': lidar_label})

        return selected_cav_processed

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']

            velocity = torch.from_numpy(np.array([cav_content['velocity']]))
            time_delay = torch.from_numpy(np.array([cav_content['time_delay']]))
            infra = torch.from_numpy(np.array([cav_content["infra"]]))
            prior_encoding = \
                torch.stack([velocity, time_delay, infra], dim=-1).float()
            record_len = torch.from_numpy(np.array([1], dtype=int))
            spatial_correction_matrix_list = \
                torch.from_numpy(np.array([cav_content['spatial_correction_matrix']]))

            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                origin_lidar = [cav_content['origin_lidar']]

            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'spatial_correction_matrix': spatial_correction_matrix_list,
                                        'prior_encoding': prior_encoding,
                                        'record_len': record_len,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'bpp_stack': cav_content['bpp_stack'],
                                        'size_stack': cav_content['size_stack'],
                                        'path': cav_content['path']})

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        spatial_correction_matrix_list = []
        record_len = []
        velocity = []
        time_delay = []
        infra = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])
            record_len.append(1)
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(ego_dict['spatial_correction_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        spatial_correction_matrix_list = \
            torch.from_numpy(np.array(spatial_correction_matrix_list))
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor, obj_id = self.post_processor.generate_gt_bbx_for_label_gen(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor, obj_id

    def generate_label_complement(self, selected_cav_base, ego_cav_base):
        if 'path' in selected_cav_base:
            cache_path = selected_cav_base['path'] + '_' + self.label_mode + '.npy'
            if os.path.exists(cache_path):
                label_complement = np.load(cache_path)
                return label_complement  # (N, 1)

        tmp_object_dict = {}
        tmp_object_dict.update(selected_cav_base['params']['vehicles'])
        transmit_vehicle = set(selected_cav_base['params']['vehicles'].keys())
        if ego_cav_base is not None:
            transmit_vehicle = transmit_vehicle - set(ego_cav_base['params']['vehicles'].keys())
        transmit_vehicle_base = {k: tmp_object_dict[k] for k in transmit_vehicle}
        other_vehicle_base = {k: tmp_object_dict[k] for k in tmp_object_dict.keys() - transmit_vehicle}

        def get_mask_for_vehicle(vehicle_base, cav_base):
            masks = [np.array([False]*cav_base['lidar_np'].shape[0])]
            for veh_id, veh_content in vehicle_base.items():
                location = veh_content['location']
                rotation = veh_content['angle']
                center = veh_content['center']
                extent = veh_content['extent']
                object_pose = [location[0] + center[0],
                               location[1] + center[1],
                               location[2] + center[2],
                               rotation[0], rotation[1], rotation[2]]
                lidar2object = x1_to_x2(cav_base['params']['lidar_pose'], object_pose)
                lidar_np = cav_base['lidar_np'][:, :3]
                lidar_np = box_utils.project_points_by_matrix_torch(lidar_np, lidar2object)  # (N, 3)
                range_veh = np.array(extent)
                mask = (lidar_np[:, 0] > -range_veh[0]) & (lidar_np[:, 0] < range_veh[0]) & \
                       (lidar_np[:, 1] > -range_veh[1]) & (lidar_np[:, 1] < range_veh[1]) & \
                       (lidar_np[:, 2] > -range_veh[2]) & (lidar_np[:, 2] < range_veh[2])
                masks.append(mask)  # (N)
            masks = np.logical_or.reduce(masks)  # (N)
            return masks

        mask_for_transmit_vehicle = get_mask_for_vehicle(transmit_vehicle_base, selected_cav_base)
        mask_for_other_vehicle = get_mask_for_vehicle(other_vehicle_base, selected_cav_base)
        if self.label_mode == 'foreground_all':
            mask_for_transmit_vehicle = np.logical_or(mask_for_transmit_vehicle, mask_for_other_vehicle)
            return mask_for_transmit_vehicle.astype(float)
        elif self.label_mode == 'bin_MSE':
            return mask_for_transmit_vehicle.astype(float)
        else:
            raise NotImplementedError
        # TODO: label save
        # TODO: other label mode

    def generate_label_complement_weight(self, selected_cav_base, ego_cav_base):
        # 生成label的同时生成权重，agent中每个物体的权重，根据其在ego视野中的点的个数而定
        if 'path' in selected_cav_base:
            cache_path = selected_cav_base['path'] + '_' + self.label_mode + '.npy'
            if os.path.exists(cache_path):
                label_complement = np.load(cache_path)
                return label_complement  # (N, 1)

        tmp_object_dict = {}
        tmp_object_dict.update(selected_cav_base['params']['vehicles'])
        transmit_vehicle = set(selected_cav_base['params']['vehicles'].keys())
        if ego_cav_base is not None:
            transmit_vehicle = transmit_vehicle - set(ego_cav_base['params']['vehicles'].keys())
        transmit_vehicle_base = {k: tmp_object_dict[k] for k in transmit_vehicle}
        other_vehicle_base = {k: tmp_object_dict[k] for k in tmp_object_dict.keys() - transmit_vehicle}

        def get_mask(veh_content, cav_base):
            location = veh_content['location']
            rotation = veh_content['angle']
            center = veh_content['center']
            extent = veh_content['extent']
            object_pose = [location[0] + center[0],
                           location[1] + center[1],
                           location[2] + center[2],
                           rotation[0], rotation[1], rotation[2]]
            lidar2object = x1_to_x2(cav_base['params']['lidar_pose'], object_pose)
            lidar_np = cav_base['lidar_np'][:, :3]
            lidar_np = box_utils.project_points_by_matrix_torch(lidar_np, lidar2object)  # (N, 3)
            range_veh = np.array(extent)
            mask = (lidar_np[:, 0] > -range_veh[0]) & (lidar_np[:, 0] < range_veh[0]) & \
                   (lidar_np[:, 1] > -range_veh[1]) & (lidar_np[:, 1] < range_veh[1]) & \
                   (lidar_np[:, 2] > -range_veh[2]) & (lidar_np[:, 2] < range_veh[2])
            return mask

        def get_mask_for_vehicle(vehicle_base, cav_base):
            masks = [np.array([False]*cav_base['lidar_np'].shape[0])]
            for veh_id, veh_content in vehicle_base.items():
                mask = get_mask(veh_content, selected_cav_base)
                masks.append(mask)  # (N)
            masks = np.logical_or.reduce(masks)  # (N)
            return masks

        # def get_mask_for_vehicle_with_weight(vehicle_base, cav_base, ego_base):
        #     masks = [np.array([False] * cav_base['lidar_np'].shape[0])]
        #     ego_veh_base = ego_base['params']['vehicles']
        #     for veh_id, veh_content in vehicle_base.items():
        #         mask = get_mask(veh_content, selected_cav_base)
        #         mask_ego = get_mask(ego_veh_base[veh_id], ego_base)
        #         masks.append(mask)  # (N)
        #     masks = np.logical_or.reduce(masks)  # (N)
        #     return masks
        # TODO: NotImplementedError

        mask_for_transmit_vehicle = get_mask_for_vehicle(transmit_vehicle_base, selected_cav_base)
        # weight_for_transmit_vehicle = np.full_like(mask_for_transmit_vehicle, 0.0)
        # weight_for_transmit_vehicle[mask_for_transmit_vehicle] = 1.0    #视野中没有出现的权重为1

        mask_for_other_vehicle = get_mask_for_vehicle(other_vehicle_base, selected_cav_base)
        if self.label_mode == 'foreground_all':
            mask_for_transmit_vehicle = np.logical_or(mask_for_transmit_vehicle, mask_for_other_vehicle)
            return mask_for_transmit_vehicle.astype(float)
        elif self.label_mode == 'bin_MSE':
            return mask_for_transmit_vehicle.astype(float)
        else:
            raise NotImplementedError
        # TODO: label save
        # TODO: other label mode

if __name__ == '__main__':
    hypes_yaml = '/home/JJ_Group/cheny/v2x-vit/v2xvit/hypes_yaml/point_pillar_early_fusion.yaml'
    hypes = yaml_utils.load_yaml(hypes_yaml)
    dataset = EarlyFusion4LabelGenerate(params=hypes, visualize=True)
    dataset.__getitem__(0)
    print('----------')

