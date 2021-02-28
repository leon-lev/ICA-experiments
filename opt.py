from functools import partial

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiplicativeLR

def initialize_optimizers(encoder, decoder, discriminator, cfg):
    ae_optim = Adam(list(encoder.parameters()) + list(decoder.parameters()),
                    lr=cfg.optim.lr_ae,
                    weight_decay=cfg.optim.weight_decay)

    disc_optim = SGD(discriminator.parameters(), 
                     lr=cfg.optim.lr_disc, 
                     weight_decay=cfg.optim.weight_decay)

    return ae_optim, disc_optim


def initalize_schedulers(ae_optim, disc_optim, cfg):
    ae_sched = MultiplicativeLR(ae_optim, lr_lambda=partial(lambda_rule_ae, cfg=cfg)) 
    disc_sched = MultiplicativeLR(disc_optim, lr_lambda=partial(lambda_rule_disc, cfg=cfg))
    return ae_sched, disc_sched


def lambda_rule_ae(epoch, cfg):
    if cfg.optim.lr_ae * (cfg.optim.gamma ** epoch) > cfg.optim.min_lr:
        return cfg.optim.gamma
    else:
        return 1

def lambda_rule_disc(epoch, cfg):
    if cfg.optim.lr_disc * (cfg.optim.gamma ** epoch) > cfg.optim.min_lr:
        return cfg.optim.gamma
    else:
        return 1