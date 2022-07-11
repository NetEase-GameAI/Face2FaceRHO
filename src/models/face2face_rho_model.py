import torch
import torch.nn.functional as F
import numpy as np
import cv2
from models import VGG19_LOSS
from models.base_model import BaseModel
from models import networks
from models.discriminator import MultiScaleDiscriminator
from models.rendering_network import RenderingNet
from models.motion_network import MotionNet
from models.image_pyramid import ImagePyramide


def define_rendering_net(opt, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = RenderingNet(opt)
    return networks.init_net(net, init_type, init_gain, gpu_ids)


def define_motion_net(opt, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = MotionNet(opt)
    return networks.init_net(net, init_type, init_gain, gpu_ids)


def define_discriminator(opt, init_type='normal', init_gain=0.02, gpu_ids=[]):
    discriminator = MultiScaleDiscriminator(scales=opt.disc_scales, num_channels=3,
                                            block_expansion=opt.disc_block_expansion, num_blocks=opt.disc_num_blocks,
                                            max_features=opt.disc_max_features, sn=False, use_kp=True)
    return networks.init_net(discriminator, init_type, init_gain, gpu_ids)


def detach_dict(kp):
    return {key: value.detach() for key, value in kp.items()}


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


class Face2FaceRHOModel(BaseModel):
    def name(self):
        return 'Face2FaceRHOModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.loss_names = ['G_total', 'warp', 'reconstruction', 'init_field', 'g_gan', 'feature_matching', 'D_total']

        if self.isTrain:
            self.visual_names = ['src_img', 'drv_img', 'fake', 'warped_src1', 'warped_src2', 'warped_src3']
            if self.opt.emphasize_face_area:
                self.visual_names = self.visual_names + ['drv_face_mask']
        else:
            self.visual_names = ['fake']

        self.model_names = ['rendering_net', 'motion_net']

        if self.opt.isTrain:
            self.model_names = self.model_names + ['discriminator']

        # load/define networks
        self.rendering_net = define_rendering_net(opt, opt.init_type, opt.init_gain, self.gpu_ids)
        self.motion_net = define_motion_net(opt, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.opt.isTrain:
            self.discriminator = define_discriminator(opt, opt.init_type, opt.init_gain, self.gpu_ids)

        # optimizer
        if self.isTrain:
            self.init_motion_field_2 = self.make_coordinate_grid(int(opt.output_size / 4), int(opt.output_size / 4))
            self.init_motion_field_1 = self.make_coordinate_grid(int(opt.output_size / 8), int(opt.output_size / 8))
            self.init_motion_field_0 = self.make_coordinate_grid(int(opt.output_size / 16), int(opt.output_size / 16))

            self.init_field_criterion = torch.nn.L1Loss(reduction='mean')
            self.warp_criterion = torch.nn.L1Loss(reduction='mean')
            self.vgg19loss = VGG19_LOSS.VGG19LOSS().to(self.device)
            self.pyramid = ImagePyramide(self.opt.loss_scales, 3).to(self.device)
            # initialize optimizers
            self.optimizers = []
            self.optimizer_rendering_net = torch.optim.Adam(self.rendering_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_motion_net = torch.optim.Adam(self.motion_net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_rendering_net)
            self.optimizers.append(self.optimizer_motion_net)
            self.optimizers.append(self.optimizer_discriminator)

    def make_coordinate_grid(self, h, w):
        """
        Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
        """
        x = torch.arange(w, dtype=torch.float32)
        y = torch.arange(h, dtype=torch.float32)
        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)
        yy = y.view(-1, 1).repeat(1, w)
        xx = x.view(1, -1).repeat(h, 1)
        meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
        meshed = meshed.unsqueeze(0)
        meshed = meshed.repeat(self.opt.batch_size, 1, 1, 1)
        meshed = meshed.permute(0, 3, 1, 2)
        meshed = meshed.to(self.device)
        return meshed

    def set_input(self, input_data):
        self.src_img = move_to(input_data['src_img'], self.device)
        self.src_headpose = move_to(input_data['src_headpose'], self.device)
        self.src_landmark_img = move_to(input_data['src_landmark_img'], self.device)
        self.drv_img = move_to(input_data['drv_img'], self.device)
        self.drv_headpose = move_to(input_data['drv_headpose'], self.device)
        self.drv_landmark_img = move_to(input_data['drv_landmark_img'], self.device)

        self.drv_face_area_weight = None
        if self.opt.emphasize_face_area:
            self.drv_face_mask = move_to(input_data['drv_face_mask'], self.device)
            self.drv_face_mask = self.drv_face_mask == 1

            # we increase the weight of the face area when calculating loss
            self.drv_face_area_weight = torch.where(
                self.drv_face_mask.unsqueeze(1),
                torch.ones_like(self.src_img[:, 0:1, ...]) * self.opt.face_area_weight_scale,
                torch.ones_like(self.src_img[:, 0:1, ...]) * 1)

    def calculate_motion_field(self):
        motion_net_input = []
        for i in range(len(self.src_landmark_img)):
            cur_input = torch.cat([self.src_landmark_img[i], self.drv_landmark_img[i]], dim=1)
            motion_net_input.append(cur_input)

        for i in range(len(motion_net_input)):
            _, _, h1, w1 = motion_net_input[i].shape
            _, _, h2, w2 = self.src_img.shape
            if h1 != h2 or w1 != w2:
                cur_src_img_input = F.interpolate(self.src_img, size=(h1, w1), mode='nearest')
            else:
                cur_src_img_input = self.src_img
            motion_net_input[i] = torch.cat([motion_net_input[i], cur_src_img_input], dim=1)

        self.motion_field = self.motion_net(motion_net_input[0], motion_net_input[1], motion_net_input[2])

    def forward(self):
        bs, _, _, _ = self.src_img.size()
        self.calculate_motion_field()
        self.fake = self.rendering_net(self.src_img, self.motion_field[2], self.src_headpose, self.drv_headpose)

    def set_source_face(self, src_img, src_headpose):
        self.src_img = move_to(src_img, self.device)
        self.src_headpose = move_to(src_headpose, self.device)
        self.rendering_net.register_source_face(self.src_img, self.src_headpose)

    def reenactment(self, src_landmark_img, drv_headpose, drv_landmark_img):
        self.src_landmark_img = move_to(src_landmark_img, self.device)
        self.drv_landmark_img = move_to(drv_landmark_img, self.device)
        self.drv_headpose = move_to(drv_headpose, self.device)

        self.calculate_motion_field()
        self.fake = self.rendering_net.reenactment(self.motion_field[2], self.drv_headpose)

    def backward_G(self):
        self.warped_src1 = Face2FaceRHOModel.deform_img(self.src_img, self.motion_field[0])
        self.warped_src2 = Face2FaceRHOModel.deform_img(self.src_img, self.motion_field[1])
        self.warped_src3 = Face2FaceRHOModel.deform_img(self.src_img, self.motion_field[2])

        self.loss_warp = 0
        self.loss_warp = self.loss_warp + self.opt.warp_loss_weight / 3 * self.calculate_warp_loss(
            self.warped_src1, self.drv_img)
        self.loss_warp = self.loss_warp + self.opt.warp_loss_weight / 3 * self.calculate_warp_loss(
            self.warped_src2, self.drv_img)
        self.loss_warp = self.loss_warp + self.opt.warp_loss_weight / 3 * self.calculate_warp_loss(
            self.warped_src3, self.drv_img)

        # To improve the stability of the training, we regularizing the warping in the first epochs to be close to
        # an identity mapping.
        self.loss_init_field = 0
        if self.cur_epoch <= self.opt.init_field_epochs:
            init_field_loss_weight = 1000 * (
                    self.opt.init_field_epochs - self.cur_epoch) / self.opt.init_field_epochs
            self.loss_init_field = self.loss_init_field + init_field_loss_weight / 3 * self.init_field_criterion(
                self.init_motion_field_2, self.motion_field[2].permute(0, 3, 1, 2))
            self.loss_init_field = self.loss_init_field + init_field_loss_weight / 3 * self.init_field_criterion(
                self.init_motion_field_1, self.motion_field[1].permute(0, 3, 1, 2))
            self.loss_init_field = self.loss_init_field + init_field_loss_weight / 3 * self.init_field_criterion(
                self.init_motion_field_0, self.motion_field[0].permute(0, 3, 1, 2))

        self.loss_reconstruction = self.opt.reconstruction_loss_weight * self.calculate_loss(
            self.vgg19loss, self.fake, self.drv_img, pyramid_scales=self.opt.loss_scales)

        self.loss_g_gan = 0

        # GAN loss
        self.generated_for_disc_use = {'prediction_1': self.fake}
        self.real_for_disc_use = {'prediction_1': self.drv_img}
        condition = self.drv_landmark_img[2]
        discriminator_maps_generated = self.discriminator(self.generated_for_disc_use, kp=condition.detach())
        discriminator_maps_real = self.discriminator(self.real_for_disc_use, kp=condition.detach())
        for scale in self.opt.disc_scales:
            key = 'prediction_map_%s' % scale
            value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
            self.loss_g_gan = self.loss_g_gan + value

        # feature matching loss
        self.loss_feature_matching = 0
        for scale in self.opt.disc_scales:
            key = 'feature_maps_%s' % scale
            for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                value = torch.abs(a - b)
                if not self.opt.emphasize_face_area:
                    value = value.mean()
                else:
                    bs, c, h1, w1 = value.shape
                    _, _, h2, w2 = self.drv_face_area_weight.shape
                    if h1 != h2 or w1 != w2:
                        cur_weight_mask = F.interpolate(self.drv_face_area_weight, size=(h1, w1))
                        value = value * cur_weight_mask
                    else:
                        value = value * self.drv_face_area_weight
                    value = value.mean()
                self.loss_feature_matching = self.loss_feature_matching + self.opt.feature_matching_loss_weight * value

        self.loss_G_total = self.loss_warp + self.loss_reconstruction + \
                            self.loss_g_gan + self.loss_feature_matching + \
                            self.loss_init_field

        self.loss_G_total.backward()

    def backward_D(self):
        condition = self.drv_landmark_img[2]
        discriminator_maps_generated = self.discriminator(
            detach_dict(self.generated_for_disc_use), kp=condition.detach())
        discriminator_maps_real = self.discriminator(self.real_for_disc_use, kp=condition.detach())

        self.loss_D_total = 0
        for scale in self.opt.disc_scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            self.loss_D_total = self.loss_D_total + value.mean()

        self.loss_D_total.backward()

    def optimize_parameters(self, epoch):
        self.cur_epoch = epoch
        self.forward()
        # update Generator
        self.optimizer_rendering_net.zero_grad()
        self.optimizer_motion_net.zero_grad()
        self.backward_G()
        self.optimizer_rendering_net.step()
        self.optimizer_motion_net.step()

        self.optimizer_discriminator.zero_grad()
        self.backward_D()
        self.optimizer_discriminator.step()

    @staticmethod
    def deform_img(inp, deformation):
        bs, h, w, _ = deformation.shape
        if h < 128 or w < 128:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(128, 128), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation)

    def calculate_loss(self, loss_criterion, input1, input2, pyramid_scales):
        _, _, h1, w1 = input1.shape
        _, _, h2, w2 = input2.shape
        if h1 != h2 or w1 != w2:
            input2 = F.interpolate(input2, size=(h1, w1), mode='bilinear')
        loss = 0
        pyramide_input1 = self.pyramid(input1)
        pyramide_input2 = self.pyramid(input2)
        for scale in pyramid_scales:
            cur_loss = loss_criterion(pyramide_input1['prediction_' + str(scale)],
                                      pyramide_input2['prediction_' + str(scale)],
                                      weight_mask=self.drv_face_area_weight)
            cur_loss = torch.mean(cur_loss)
            loss = loss + cur_loss
        return loss

    def calculate_warp_loss(self, input1, input2):
        _, _, h1, w1 = input1.shape
        _, _, h2, w2 = input2.shape
        if h1 != h2 or w1 != w2:
            input2 = F.interpolate(input2, size=(h1, w1), mode='bilinear')
        return self.warp_criterion(input1, input2)
