import torch.nn.functional as F
from torch import nn


class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=2)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1, use_relu=True,
                 sample_mode='nearest'):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.use_relu = use_relu
        self.sample_mode = sample_mode

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2, mode=self.sample_mode)
        out = self.conv(out)
        out = self.norm(out)
        if self.use_relu:
            out = F.relu(out)
        return out


class ResBlock(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = nn.BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class MotionNet(nn.Module):
    def __init__(self, opt):
        super(MotionNet, self).__init__()

        ngf = opt.mn_ngf
        n_local_enhancers = opt.n_local_enhancers
        n_downsampling = opt.mn_n_downsampling
        n_blocks_local = opt.mn_n_blocks_local

        in_features = [9, 9, 9]

        # F1
        f1_model_ngf = ngf * (2 ** n_local_enhancers)
        f1_model = [
            nn.Conv2d(in_channels=in_features[0], out_channels=f1_model_ngf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1_model_ngf),
            nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            f1_model += [
                DownBlock(f1_model_ngf * mult, f1_model_ngf * mult * 2, kernel_size=4, padding=1, use_relu=True)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            f1_model += [
                UpBlock(f1_model_ngf * mult, int(f1_model_ngf * mult / 2), kernel_size=3, padding=1)
            ]

        self.f1_model = nn.Sequential(*f1_model)
        self.f1_motion = nn.Conv2d(f1_model_ngf, 2, kernel_size=(3, 3), padding=(1, 1))

        #f2 and f3
        for n in range(1, n_local_enhancers + 1):
            ### first downsampling block
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_first_downsample = [DownBlock(in_features[n], ngf_global * 2, kernel_size=4, padding=1, use_relu=True)]
            ### other downsampling blocks, residual blocks and upsampling blocks
            # other downsampling blocks
            model_other = []
            model_other += [
                DownBlock(ngf_global * 2, ngf_global * 4, kernel_size=4, padding=1, use_relu=True),
                DownBlock(ngf_global * 4, ngf_global * 8, kernel_size=4, padding=1, use_relu=True),
            ]
            # residual blocks
            for i in range(n_blocks_local):
                model_other += [ResBlock(ngf_global * 8, 3, 1)]
            # upsampling blocks
            model_other += [
                UpBlock(ngf_global * 8, ngf_global * 4, kernel_size=3, padding=1),
                UpBlock(ngf_global * 4, ngf_global * 2, kernel_size=3, padding=1),
                UpBlock(ngf_global * 2, ngf_global, kernel_size=3, padding=1)
            ]
            model_motion = nn.Conv2d(ngf_global, out_channels=2, kernel_size=3, padding=1, groups=1)

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_first_downsample))
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_other))
            setattr(self, 'model' + str(n) + '_3', model_motion)

    def forward(self, input1, input2, input3):
        ### output at small scale(f1)
        output_prev = self.f1_model(input1)
        low_motion = self.f1_motion(output_prev)

        ### output at middle scale(f2)
        output_prev = self.model1_2(self.model1_1(input2) + output_prev)
        middle_motion = self.model1_3(output_prev)
        middle_motion = middle_motion + nn.Upsample(scale_factor=2, mode='nearest')(low_motion)

        ### output at large scale(f3)
        output_prev = self.model2_2(self.model2_1(input3) + output_prev)
        high_motion = self.model2_3(output_prev)
        high_motion = high_motion + nn.Upsample(scale_factor=2, mode='nearest')(middle_motion)

        low_motion = low_motion.permute(0, 2, 3, 1)
        middle_motion = middle_motion.permute(0, 2, 3, 1)
        high_motion = high_motion.permute(0, 2, 3, 1)
        return [low_motion, middle_motion, high_motion]
