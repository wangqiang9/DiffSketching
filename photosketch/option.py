import argparse
import torch

save = './save_photosketch'


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # change options
        self.parser.add_argument('--name', type=str, default='pretrained',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='test_dir',
                                 help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--dataroot', default='./examples/',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--results_dir', type=str, default=f'./{save}/Results', help='saves results here.')
        self.parser.add_argument('--checkpoints_dir', type=str, default=f'./{save}/Checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='pix2pix',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks',
                                 help='selects model to use for netG')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--pretrain_path', type=str, default='./pre_model')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')



        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='disable CUDA training (please use CUDA_VISIBLE_DEVICES to select GPU)')
        self.parser.add_argument('--serial_batches', action='store_true', default=True,
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost",
                                 help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', default=True,
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--render_dir', type=str, default='sketch-rendered')
        self.parser.add_argument('--aug_folder', type=str, default='width-5')
        self.parser.add_argument('--stroke_dir', type=str, default='')
        self.parser.add_argument('--crop', action='store_true')
        self.parser.add_argument('--rotate', action='store_true')
        self.parser.add_argument('--color_jitter', action='store_true')
        self.parser.add_argument('--stroke_no_couple', action='store_true', help='')
        self.parser.add_argument('--nGT', type=int, default=5)
        self.parser.add_argument('--rot_int_max', type=int, default=3)
        self.parser.add_argument('--jitter_amount', type=float, default=0.02)
        self.parser.add_argument('--inverse_gamma', action='store_true')
        self.parser.add_argument('--img_mean', type=float, nargs='+')
        self.parser.add_argument('--img_std', type=float, nargs='+')
        self.parser.add_argument('--lst_file', type=str)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        self.opt.use_cuda = not self.opt.no_cuda and torch.cuda.is_available()

        return self.opt


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--file_name', type=str, default='')
        self.parser.add_argument('--suffix', type=str, default='')
        self.isTrain = False

if __name__ == "__main__":
    opt = TestOptions().parse()