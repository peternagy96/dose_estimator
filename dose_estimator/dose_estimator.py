# !/usr/bin/env python3
from models.gan import cycleGAN
from trainer import Trainer

if __name__ == '__main__':
    # import model
    gan = cycleGAN()

    # load trainer
    trainer = Trainer()

    gan.D_A.compile(optimizer=self.opt_D,
                    loss=self.lse,
                    loss_weights=loss_weights)
    gan.D_B.compile(optimizer=self.opt_D,
                    loss=self.lse,
                    loss_weights=gan.D_B.loss_weights)

    if self.use_identity_learning:
        gan.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
        gan.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

    # decide on mode

    # run training

    # run testing

    # run test on NIFTI

    # plot model
