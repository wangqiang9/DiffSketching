
def create_model(opt):
    model = None
    if opt.model == 'pix2pix':
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    return model
