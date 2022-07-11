# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
import torch.nn as nn
from .models.encoders import ResnetEncoder
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.config import cfg
torch.backends.cudnn.benchmark = True

class DECA(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    # @torch.no_grad()
    def encode(self, images, use_detail=False):
        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose  
        return codedict

    def ensemble_3DMM_params(self, codedict, image_size, original_image_size):
        i = 0
        cam = codedict['cam']
        tform = codedict['tform']
        scale, tx, ty, sz = util.calculate_scale_tx_ty(cam, tform, image_size, original_image_size)
        crop_scale, crop_tx, crop_ty, crop_sz = util.calculate_crop_scale_tx_ty(cam)
        scale = float(scale[i].cpu())
        tx = float(tx[i].cpu())
        ty = float(ty[i].cpu())
        sz = float(sz[i].cpu())

        crop_scale = float(crop_scale[i].cpu())
        crop_tx = float(crop_tx[i].cpu())
        crop_ty = float(crop_ty[i].cpu())
        crop_sz = float(crop_sz[i].cpu())

        shape_params = codedict['shape'][i].cpu().numpy()
        expression_params = codedict['exp'][i].cpu().numpy()
        pose_params = codedict['pose'][i].cpu().numpy()

        face_model_paras = dict()
        face_model_paras['shape'] = shape_params.tolist()
        face_model_paras['exp'] = expression_params.tolist()
        face_model_paras['pose'] = pose_params.tolist()
        face_model_paras['cam'] = cam[i].cpu().numpy().tolist()

        face_model_paras['scale'] = scale
        face_model_paras['tx'] = tx
        face_model_paras['ty'] = ty
        face_model_paras['sz'] = sz

        face_model_paras['crop_scale'] = crop_scale
        face_model_paras['crop_tx'] = crop_tx
        face_model_paras['crop_ty'] = crop_ty
        face_model_paras['crop_sz'] = crop_sz
        return face_model_paras
