import numpy as np
import torch
import os
from .base_model import BaseModel
from . import networks
from PIL import Image

save = './save_photosketch'
class OPT(object):
    name = "pretrained"
    dataset_mode = "test_dir"
    dataroot = "./examples/"
    results_dir = f'./{save}/Results'
    checkpoints_dir = f'./{save}/Checkpoints'
    model = "pix2pix"
    which_direction = "AtoB"
    norm = "batch"
    input_nc = 3
    output_nc = 1
    which_model_netG = "resnet_9blocks"
    no_dropout = "store_true"
    pretrain_path = "./models"
    nThreads = 1
    batchSize = 1
    loadSize = 286
    fineSize = 256
    ngf = 64
    ndf = 64
    which_model_netD = "basic"
    n_layers_D = 3
    use_cuda = torch.cuda.is_available()
    serial_batches = True
    display_winsize = 256
    display_id = 1
    display_server = "http://localhost"
    display_port = 8097
    max_dataset_size = float("inf")
    resize_or_crop = "resize_and_crop"
    no_flip = True
    init_type = "normal"
    render_dir = "sketch-rendered"
    aug_folder = "width-5"
    stroke_dir = ""
    nGT = 5
    rot_int_max = 3
    jitter_amount = 0.02
    ntest = float("inf")
    aspect_ratio = 1.0
    phase = "test"
    which_epoch = "latest"
    how_many = 50
    file_name = ""
    suffix = ""
    isTrain = False

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        self.netG = self.netG.to(self.device)

    def set_input(self, input):
        self.input_A = input.to(self.device)

    def forward(self):
        self.real_A = self.input_A
        self.fake_B = self.netG(self.real_A)
        # self.real_B = self.input_B

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.forward()
        return self.fake_B

    def save(self, label):
        self.save_network(self.netG, 'G', label)
        self.save_network(self.netD, 'D', label)


    def write_image(self, out_dir, name):
        image_numpy = self.fake_B.detach()[0][0].cpu().float().numpy()
        image_pil = Image.fromarray(image_numpy.astype(np.uint8))
        print(image_pil.size)
        out_path = os.path.join(out_dir, name + self.opt.suffix + '.png')
        image_pil.save(out_path)
       