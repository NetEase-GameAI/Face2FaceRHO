import argparse
import time
from options.parse_config import Face2FaceRHOConfigParse
from dataset import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os


def parse_args():
    """Configurations."""
    parser = argparse.ArgumentParser(description='training process of Face2FaceRHO')
    parser.add_argument('--config', type=str, required=True, help='.ini config file name')
    return _check_args(parser.parse_args())


def _check_args(args):
    if args is None:
        raise RuntimeError('Invalid arguments!')
    return args


if __name__ == '__main__':
    print(os.getcwd())
    args = parse_args()
    config_parse = Face2FaceRHOConfigParse()
    opt = config_parse.get_opt_from_ini(args.config)  # get training options
    config_parse.setup_environment()

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_steps = 0

    display_inter = -1
    print_inter = -1
    save_latest_inter = -1

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0  # iterator within an epoch

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters(epoch)
            print("total steps: {}".format(total_steps))

            if total_steps // opt.display_freq > display_inter:
                display_inter = total_steps // opt.display_freq
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, False)

            if total_steps // opt.print_freq > print_inter:
                print_inter = total_steps // opt.print_freq
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps // opt.save_latest_freq > save_latest_inter:
                save_latest_inter = total_steps // opt.save_latest_freq
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                # model.save_networks('latest')
                save_suffix = 'iter_%d' % total_steps
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()