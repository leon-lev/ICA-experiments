from yacs.config import CfgNode as CN

cfg = CN()
cfg.device = 'cuda'
cfg.save_dir = './runs'
# spatial size of input images
cfg.in_size = 256
# Number of input channels (RGB)
cfg.in_channels = 3
# Number of input classes
cfg.n_classes = 5
# independence loss coefficient
cfg.beta = 0.02
# number of feature channels of latent representation
cfg.code_channels = 4
# training batch size
cfg.batch_size = 16
# training epochs
cfg.n_epochs = 5
# dataset folder
cfg.dataset_dir = 'datasets/art'

cfg.optim = CN()
# learning rate autoencoder
cfg.optim.lr_ae = 1e-5
# learning rate disctiminator
cfg.optim.lr_disc = 1e-4
# weight_decay
cfg.optim.weight_decay = 1e-5
# LR scheduler multiplicative factor
cfg.optim.gamma = 0.97
# min value for LR scheduler
cfg.optim.min_lr = 1e-5

cfg.autoencoder = CN()
# the period (in number of batches) for autoencoder optimization
cfg.autoencoder.opt_interval = 5
# activation function at the final layer
cfg.autoencoder.decoder_activation = 'sigmoid'


cfg.discriminator = CN()
# the period (in number of batches) for discriminator optimization
cfg.discriminator.opt_interval = 1

cfg.visualization = CN()
# whether to show model output visualization
cfg.visualization.show = True
# how often model outputs should be visualized during training
cfg.visualization.epoch_interval = 10
# number of examples in visualization
cfg.visualization.n_examples = 5