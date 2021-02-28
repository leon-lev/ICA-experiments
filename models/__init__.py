import logging

import torch

from .autoencoder import UNetModel
from .discriminator import Discriminator

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def initialize_models(cfg):
    logging.info('Loading encoder ...')
    encoder = UNetModel(n_channels_in=cfg.in_channels,
                        n_channels_out=cfg.code_channels,
                        n_classes=cfg.n_classes)
    
    logging.info('Loading decoder ...')
    decoder = UNetModel(n_channels_in=cfg.code_channels,
                        n_channels_out=cfg.in_channels,
                        n_classes=cfg.n_classes,
                        output_activation=cfg.autoencoder.decoder_activation)
    
    logging.info('Loading discriminator ...')
    discriminator = Discriminator(n_channels_in=cfg.code_channels,
                                        n_classes=cfg.n_classes)
    logging.info('Finished loading models')
    return encoder, decoder, discriminator

def validate_models(encoder, decoder, discriminator, cfg):
    batch_size = 4
    device = cfg.device
    if not torch.cuda.is_available():
        assert cfg.device == 'cpu', 'CUDA unabailable, set cfg.device to "cpu"'
    
    encoder.to(device)
    decoder.to(device)
    discriminator.to(device)
    
    img = torch.rand(batch_size,
                        cfg.in_channels,
                        cfg.in_size,
                        cfg.in_size).to(device)
    
    cond = torch.randint(high=cfg.n_classes,
                            size=(batch_size,)).to(device)

    with torch.no_grad():
        code = encoder(img, cond)
        print(f'Code shape {code.shape}')
        recon = decoder(code, cond)
        print(f'Reconstuction shape {recon.shape}')
        logits = discriminator(code)
        print(f'Logits shape {logits.shape}')
        
        assert code.shape == (batch_size,
                                cfg.code_channels,
                                cfg.in_size,
                                cfg.in_size,), "code shape don't match"
        
        assert recon.shape == (batch_size,
                                cfg.in_channels,
                                cfg.in_size,
                                cfg.in_size,), "reconstruction shape don't match"

        assert logits.shape == (batch_size,
                                cfg.n_classes), "logits shape don't match"