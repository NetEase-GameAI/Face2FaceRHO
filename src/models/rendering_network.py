import torch.nn.functional as F
from torch import nn
import torch


class PoseEncoder(nn.Module):
    def __init__(self, ngf, headpose_dims):
        super(PoseEncoder, self).__init__()
        self.ngf = ngf
        self.embedding_module1 = nn.Sequential(
            nn.ConvTranspose2d(headpose_dims, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        )
        self.embedding_module2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 4, 0, bias=False),
        )

    def get_embedding_feature_map_channel(self):
        return self.ngf * 4

    def forward(self, headpose):
        bs, dim = headpose.size()
        cur_embedding = self.embedding_module1(headpose.view(bs, dim, 1, 1))
        cur_embedding = self.embedding_module2(cur_embedding)
        return cur_embedding


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, final_use_norm=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        norm = nn.BatchNorm2d

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if self.use_res_connect:
            final_use_norm = True

        if expand_ratio == 1:
            conv = [
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                norm(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            ]
            if final_use_norm:
                conv += [norm(oup)]
        else:
            conv = [
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                norm(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                norm(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            ]
            if final_use_norm:
                conv += [norm(oup)]

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SPADE(nn.Module):
    def __init__(self, input_channel, label_nc):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(input_channel, affine=False)

        nhidden = label_nc * 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, input_channel, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, input_channel, kernel_size=3, padding=1)

    def forward(self, x, condition_map):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        _, c1, h1, w1 = x.size()
        _, c2, h2, w2 = condition_map.size()
        if h1 != h2 or w1 != w2:
            raise ValueError('x and condition_map have different sizes.')
        actv = self.mlp_shared(condition_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super(RenderingNet, self).__init__()

        self.src_headpose_encoder = PoseEncoder(headpose_dims=opt.headpose_dims, ngf=opt.headpose_embedding_ngf)
        self.headpose_feature_cn = self.src_headpose_encoder.get_embedding_feature_map_channel()

        self.drv_headpose_encoder = PoseEncoder(headpose_dims=opt.headpose_dims, ngf=opt.headpose_embedding_ngf)

        norm = nn.BatchNorm2d

        # define block in E_I
        def encoder_block(inc, ouc, t, n, s, final_use_norm=True):
            model = []
            input_channel = int(inc)
            output_channel = int(ouc)
            for i in range(n):
                if i == 0:
                    if n > 1:
                        model.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t))
                    else:
                        model.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t, final_use_norm=final_use_norm))
                else:
                    model.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, final_use_norm=final_use_norm))
                input_channel = output_channel

            return nn.Sequential(nn.Sequential(*model))

        # define block in D_I
        def decoder_block(inc, ouc, t, n, s, final_use_norm=True):
            model = []
            input_channel = int(inc)
            output_channel = int(ouc)
            for i in range(n):
                model.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, final_use_norm=final_use_norm))
                input_channel = output_channel
            if s == 2:
                model.append(nn.Upsample(scale_factor=2, mode='nearest'))

            return nn.Sequential(nn.Sequential(*model))

        en_channels = opt.mobilev2_encoder_channels
        de_channels = opt.mobilev2_decoder_channels
        en_layers = opt.mobilev2_encoder_layers
        de_layers = opt.mobilev2_decoder_layers
        en_expansion_factor = opt.mobilev2_encoder_expansion_factor
        de_expansion_factor = opt.mobilev2_decoder_expansion_factor

        self.en_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=en_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            norm(en_channels[0]),
            nn.ReLU(True),
        )

        self.en_down_block1 = nn.Sequential(
            encoder_block(t=en_expansion_factor[0], ouc=en_channels[1], n=en_layers[0], s=1, inc=en_channels[0]),
            encoder_block(t=en_expansion_factor[1], ouc=en_channels[2], n=en_layers[1], s=2, inc=en_channels[1]),
            encoder_block(t=en_expansion_factor[2], ouc=en_channels[3], n=en_layers[2], s=2, inc=en_channels[2],
                          final_use_norm=False),
        )

        self.en_SPADE1 = SPADE(en_channels[3], self.headpose_feature_cn)
        self.en_SPADE1_act = nn.ReLU(True)

        self.en_down_block2 = nn.Sequential(
            encoder_block(t=en_expansion_factor[3], ouc=en_channels[4], n=en_layers[3], s=2, inc=en_channels[3]),
            encoder_block(t=en_expansion_factor[4], ouc=en_channels[5], n=en_layers[4], s=1, inc=en_channels[4],
                          final_use_norm=False),
        )

        self.en_SPADE_2 = SPADE(en_channels[5], self.headpose_feature_cn)
        self.en_SPADE_2_act = nn.ReLU(True)

        self.en_down_block3 = nn.Sequential(
            encoder_block(t=en_expansion_factor[5], ouc=en_channels[6], n=en_layers[5], s=2, inc=en_channels[5],
                          final_use_norm=False),
        )

        self.en_SPADE_3 = SPADE(en_channels[6], self.headpose_feature_cn)
        self.en_SPADE_3_act = nn.ReLU(True)

        self.en_res_block = nn.Sequential(
            encoder_block(t=en_expansion_factor[6], ouc=en_channels[7], n=en_layers[6], s=1, inc=en_channels[6],
                          final_use_norm=False),
        )

        self.en_SPADE_4 = SPADE(en_channels[7], self.headpose_feature_cn)
        self.en_SPADE_4_act = nn.ReLU(True)


        self.de_SPADE_1 = SPADE(de_channels[7], self.headpose_feature_cn)
        self.de_SPADE_1_act = nn.ReLU(True)
        self.de_res_block = nn.Sequential(
            decoder_block(t=de_expansion_factor[6], ouc=de_channels[6], n=de_layers[6], s=1, inc=en_channels[7],
                          final_use_norm=False)
        )

        self.de_SPADE_2 = SPADE(de_channels[6] + en_channels[6], self.headpose_feature_cn)
        self.de_SPADE_2_act = nn.ReLU(True)
        self.de_up_block1 = nn.Sequential(
            decoder_block(t=de_expansion_factor[5], ouc=de_channels[5], n=de_layers[5], s=2,
                          inc=de_channels[6] + en_channels[6]),
        )

        self.de_SPADE_3 = SPADE(de_channels[5] + en_channels[5], self.headpose_feature_cn)
        self.de_SPADE_3_act = nn.ReLU(True)
        self.de_up_block2 = nn.Sequential(
            decoder_block(t=de_expansion_factor[4], ouc=de_channels[4], n=de_layers[4], s=1,
                          inc=de_channels[5] + en_channels[5]),
            decoder_block(t=de_expansion_factor[3], ouc=de_channels[3], n=de_layers[3], s=2, inc=de_channels[4]),
        )

        self.de_SPADE_4 = SPADE(de_channels[3] + en_channels[3], self.headpose_feature_cn)
        self.de_SPADE_4_act = nn.ReLU(True)
        self.de_up_block3 = nn.Sequential(
            decoder_block(t=de_expansion_factor[2], ouc=de_channels[2], n=de_layers[2], s=2,
                          inc=de_channels[3] + en_channels[3]),
            decoder_block(t=de_expansion_factor[1], ouc=de_channels[1], n=de_layers[1], s=2, inc=de_channels[2]),
            decoder_block(t=de_expansion_factor[0], ouc=de_channels[0], n=de_layers[0], s=1, inc=de_channels[1]),
        )

        self.de_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=de_channels[0], out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def deform_input(inp, deformation):
        bs, h1, w1, _ = deformation.shape
        bs, c, h2, w2 = inp.shape
        if h1 != h2 or w1 != w2:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h2, w2), mode='nearest')
            deformation = deformation.permute(0, 2, 3, 1)
        trans_feature = F.grid_sample(inp, deformation)
        return trans_feature

    @staticmethod
    def resize_headpose_embedding(inp, embedding):
        bs, c, h1, w1 = inp.shape
        _, _, h2, w2 = embedding.shape
        if h1 != h2 or w1 != w2:
            embedding = F.interpolate(embedding, size=(h1, w1), mode='nearest')
        return embedding

    def forward(self, src_img, motion_field, src_headpose, drv_headpose):
        # encode
        src_headpose_embedding = self.src_headpose_encoder(src_headpose)

        x = self.en_conv_block(src_img)
        x1 = self.en_down_block1(x)
        x2_in = self.en_SPADE1(x1, self.resize_headpose_embedding(x1, src_headpose_embedding))
        x2_in = nn.ReLU(True)(x2_in)

        x2 = self.en_down_block2(x2_in)
        x3_in = self.en_SPADE_2(x2, self.resize_headpose_embedding(x2, src_headpose_embedding))
        x3_in = nn.ReLU(True)(x3_in)

        x3 = self.en_down_block3(x3_in)
        x4_in = self.en_SPADE_3(x3, self.resize_headpose_embedding(x3, src_headpose_embedding))
        x4_in = nn.ReLU(True)(x4_in)

        x4 = self.en_res_block(x4_in)
        de_x4_in = self.en_SPADE_4(x4, self.resize_headpose_embedding(x4, src_headpose_embedding))
        de_x4_in = nn.ReLU(True)(de_x4_in)

        # feature warping
        trans_features = []
        trans_features.append(RenderingNet.deform_input(x2_in, motion_field))
        trans_features.append(RenderingNet.deform_input(x3_in, motion_field))
        trans_features.append(RenderingNet.deform_input(x4_in, motion_field))
        trans_features.append(RenderingNet.deform_input(de_x4_in, motion_field))

        #decode
        drv_headpose_embedding = self.drv_headpose_encoder(drv_headpose)
        x4_in = self.de_SPADE_1(trans_features[-1], self.resize_headpose_embedding(trans_features[-1], drv_headpose_embedding))
        x4_in = nn.ReLU(True)(x4_in)
        x4_out = self.de_res_block(x4_in)

        x3_in = torch.cat([x4_out, trans_features[-2]], dim=1)
        x3_in = self.de_SPADE_2(x3_in, self.resize_headpose_embedding(x3_in, drv_headpose_embedding))
        x3_in = nn.ReLU(True)(x3_in)
        x3_out = self.de_up_block1(x3_in)

        x2_in = torch.cat([x3_out, trans_features[-3]], dim=1)
        x2_in = self.de_SPADE_3(x2_in, self.resize_headpose_embedding(x2_in, drv_headpose_embedding))
        x2_in = nn.ReLU(True)(x2_in)
        x2_out = self.de_up_block2(x2_in)

        x1_in = torch.cat([x2_out, trans_features[-4]], dim=1)
        x1_in = self.de_SPADE_4(x1_in, self.resize_headpose_embedding(x1_in, drv_headpose_embedding))
        x1_in = nn.ReLU(True)(x1_in)
        x1_out = self.de_up_block3(x1_in)
        x_out = self.de_conv_block(x1_out)
        return x_out

    def register_source_face(self, src_img, src_headpose):
        # encode
        src_headpose_embeddings = self.src_headpose_encoder(src_headpose)
        x = self.en_conv_block(src_img)
        x1 = self.en_down_block1(x)
        x2_in = self.en_SPADE1(x1, self.resize_headpose_embedding(x1, src_headpose_embeddings))
        self.x2_in = nn.ReLU(True)(x2_in)

        x2 = self.en_down_block2(x2_in)
        x3_in = self.en_SPADE_2(x2, self.resize_headpose_embedding(x2, src_headpose_embeddings))
        self.x3_in = nn.ReLU(True)(x3_in)

        x3 = self.en_down_block3(x3_in)
        x4_in = self.en_SPADE_3(x3, self.resize_headpose_embedding(x3, src_headpose_embeddings))
        self.x4_in = nn.ReLU(True)(x4_in)

        x4 = self.en_res_block(x4_in)
        de_x4_in = self.en_SPADE_4(x4, self.resize_headpose_embedding(x4, src_headpose_embeddings))
        self.de_x4_in = nn.ReLU(True)(de_x4_in)

    def reenactment(self, motion_field, drv_headpose):
        # feature warping
        trans_features = []
        trans_features.append(RenderingNet.deform_input(self.x2_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.x3_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.x4_in, motion_field))
        trans_features.append(RenderingNet.deform_input(self.de_x4_in, motion_field))

        # decode
        drv_headpose_embeddings = self.drv_headpose_encoder(drv_headpose)
        x4_in = self.de_SPADE_1(trans_features[-1], self.resize_headpose_embedding(trans_features[-1], drv_headpose_embeddings))
        x4_in = nn.ReLU(True)(x4_in)
        x4_out = self.de_res_block(x4_in)

        x3_in = torch.cat([x4_out, trans_features[-2]], dim=1)
        x3_in = self.de_SPADE_2(x3_in, self.resize_headpose_embedding(x3_in, drv_headpose_embeddings))
        x3_in = nn.ReLU(True)(x3_in)
        x3_out = self.de_up_block1(x3_in)

        x2_in = torch.cat([x3_out, trans_features[-3]], dim=1)
        x2_in = self.de_SPADE_3(x2_in, self.resize_headpose_embedding(x2_in, drv_headpose_embeddings))
        x2_in = nn.ReLU(True)(x2_in)
        x2_out = self.de_up_block2(x2_in)

        x1_in = torch.cat([x2_out, trans_features[-4]], dim=1)
        x1_in = self.de_SPADE_4(x1_in, self.resize_headpose_embedding(x1_in, drv_headpose_embeddings))
        x1_in = nn.ReLU(True)(x1_in)
        x1_out = self.de_up_block3(x1_in)
        x_out = self.de_conv_block(x1_out)
        return x_out



