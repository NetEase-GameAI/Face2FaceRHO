import configparser
from util import util
import os
import torch
from abc import ABC


class Options():
    def __init__(self):
        pass


def _get_value_from_ini(conf, section, option, type=str, default=None):
    if conf.has_option(section, option):
        if type == bool:
            return conf.get(section, option) == str(True)
        else:
            return type(conf.get(section, option))
    else:
        return default


def str2ids(input_str):
    str_ids = input_str.split(',')
    ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            ids.append(id)
    return ids


def str2floats(input_str):
    str_ids = input_str.split(',')
    ids = []
    for str_id in str_ids:
        id = float(str_id)
        if id >= 0:
            ids.append(id)
    return ids


def str2ints(input_str):
    str_ids = input_str.split(',')
    ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            ids.append(id)
    return ids


class ConfigParse(ABC):
    def __init__(self):
        self.conf = configparser.ConfigParser()
        self.opt = 0

    @staticmethod
    def get_opt_from_ini(self, file_name):
        return self.opt

    def setup_environment(self):
        # print options
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(self.opt).items():
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

    def setup_test_environment(self):
        # print options
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(self.opt).items():
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])


class Face2FaceRHOConfigParse(ConfigParse):
    def __init__(self):
        ConfigParse.__init__(self)

    def get_opt_from_ini(self, file_name):
        self.conf.read(file_name, encoding="utf-8")
        opt = Options()
        # basic config
        opt.name = self.conf.get("ROOT", "name")
        opt.gpu_ids = self.conf.get("ROOT", "gpu_ids")
        opt.gpu_ids = str2ids(opt.gpu_ids)
        opt.checkpoints_dir = self.conf.get("ROOT", "checkpoints_dir")
        opt.model = self.conf.get("ROOT", "model")
        opt.output_size = int(self.conf.get("ROOT", "output_size"))
        opt.isTrain = self.conf.get("ROOT", "isTrain") == 'True'
        opt.phase = self.conf.get("ROOT", "phase")
        opt.load_iter = int(self.conf.get("ROOT", "load_iter"))
        opt.epoch = int(self.conf.get("ROOT", "epoch"))

        # rendering module config
        opt.headpose_dims = int(self.conf.get("ROOT", "headpose_dims"))
        opt.mobilev2_encoder_channels = self.conf.get("ROOT", "mobilev2_encoder_channels")
        opt.mobilev2_encoder_channels = str2ints(opt.mobilev2_encoder_channels)
        opt.mobilev2_decoder_channels = self.conf.get("ROOT", "mobilev2_decoder_channels")
        opt.mobilev2_decoder_channels = str2ints(opt.mobilev2_decoder_channels)
        opt.mobilev2_encoder_layers = self.conf.get("ROOT", "mobilev2_encoder_layers")
        opt.mobilev2_encoder_layers = str2ints(opt.mobilev2_encoder_layers)
        opt.mobilev2_decoder_layers = self.conf.get("ROOT", "mobilev2_decoder_layers")
        opt.mobilev2_decoder_layers = str2ints(opt.mobilev2_decoder_layers)
        opt.mobilev2_encoder_expansion_factor = self.conf.get("ROOT", "mobilev2_encoder_expansion_factor")
        opt.mobilev2_encoder_expansion_factor = str2ints(opt.mobilev2_encoder_expansion_factor)
        opt.mobilev2_decoder_expansion_factor = self.conf.get("ROOT", "mobilev2_decoder_expansion_factor")
        opt.mobilev2_decoder_expansion_factor = str2ints(opt.mobilev2_decoder_expansion_factor)
        opt.headpose_embedding_ngf = int(self.conf.get("ROOT", "headpose_embedding_ngf"))

        # motion module config
        opt.mn_ngf = int(self.conf.get("ROOT", "mn_ngf"))
        opt.n_local_enhancers = int(self.conf.get("ROOT", "n_local_enhancers"))
        opt.mn_n_downsampling = int(self.conf.get("ROOT", "mn_n_downsampling"))
        opt.mn_n_blocks_local = int(self.conf.get("ROOT", "mn_n_blocks_local"))

        # discriminator
        opt.disc_scales = [1]
        opt.disc_block_expansion = int(self.conf.get("ROOT", "disc_block_expansion"))
        opt.disc_num_blocks = int(self.conf.get("ROOT", "disc_num_blocks"))
        opt.disc_max_features = int(self.conf.get("ROOT", "disc_max_features"))

        # training parameters
        opt.init_type = self.conf.get("ROOT", "init_type")
        opt.init_gain = float(self.conf.get("ROOT", "init_gain"))
        opt.emphasize_face_area = self.conf.get("ROOT", "emphasize_face_area") == 'True'
        opt.loss_scales = self.conf.get("ROOT", "loss_scales")
        opt.loss_scales = str2floats(opt.loss_scales)
        opt.warp_loss_weight = float(self.conf.get("ROOT", "warp_loss_weight"))
        opt.reconstruction_loss_weight = float(self.conf.get("ROOT", "reconstruction_loss_weight"))
        opt.feature_matching_loss_weight = float(self.conf.get("ROOT", "feature_matching_loss_weight"))
        opt.face_area_weight_scale = float(self.conf.get("ROOT", "face_area_weight_scale"))
        opt.init_field_epochs = int(self.conf.get("ROOT", "init_field_epochs"))
        opt.lr = float(self.conf.get("ROOT", "lr"))
        opt.beta1 = float(self.conf.get("ROOT", "beta1"))
        opt.lr_policy = self.conf.get("ROOT", "lr_policy")
        opt.epoch_count = int(self.conf.get("ROOT", "epoch_count"))
        opt.niter = int(self.conf.get("ROOT", "niter"))
        opt.niter_decay = int(self.conf.get("ROOT", "niter_decay"))
        opt.continue_train = self.conf.get("ROOT", "continue_train") == 'True'

        # dataset parameters
        opt.dataset_mode = self.conf.get("ROOT", "dataset_mode")
        opt.dataroot = self.conf.get("ROOT", "dataroot")
        opt.num_repeats = int(self.conf.get("ROOT", "num_repeats"))
        opt.batch_size = int(self.conf.get("ROOT", "batch_size"))
        opt.serial_batches = self.conf.get("ROOT", "serial_batches") == 'True'
        opt.num_threads = int(self.conf.get("ROOT", "num_threads"))
        opt.max_dataset_size = float("inf")

        # vis_config
        opt.display_freq = int(self.conf.get("ROOT", "display_freq"))
        opt.update_html_freq = int(self.conf.get("ROOT", "update_html_freq"))
        opt.display_id = int(self.conf.get("ROOT", "display_id"))
        opt.display_server = self.conf.get("ROOT", "display_server")
        opt.display_env = self.conf.get("ROOT", "display_env")
        opt.display_port = int(self.conf.get("ROOT", "display_port"))
        opt.print_freq = int(self.conf.get("ROOT", "print_freq"))
        opt.save_latest_freq = int(self.conf.get("ROOT", "save_latest_freq"))
        opt.save_epoch_freq = int(self.conf.get("ROOT", "save_epoch_freq"))
        opt.no_html = self.conf.get("ROOT", "no_html") == str(True)
        opt.display_winsize = int(self.conf.get("ROOT", "display_winsize"))
        opt.display_ncols = int(self.conf.get("ROOT", "display_ncols"))
        opt.verbose = self.conf.get("ROOT", "verbose") == 'True'
        self.opt = opt
        return self.opt

