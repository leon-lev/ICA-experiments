import argparse
import logging

import numpy as np

from config import cfg
from data import get_loader
from ica import ICAModel
from models import initialize_models, validate_models


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def parse_opts():
    parser = argparse.ArgumentParser(description="Path to config file")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg


# if __name__ == '__main__':
cfg = parse_opts()
logging.info(f'Training arguments:\n{cfg}')

train_loader, train_dataset, class_counts = get_loader(cfg, 'train')
class_weights = np.array(class_counts) / sum(class_counts)
val_loader, val_dataset, _ = get_loader(cfg, 'val')

encoder, decoder, discriminator = initialize_models(cfg)
validate_models(encoder, decoder, discriminator, cfg)
logging.info('Model initialization')
model = ICAModel(encoder, decoder, discriminator, cfg)
logging.info('Training ... ')
model.train(train_loader, val_loader, cfg.n_epochs, vis_dataset=val_dataset)
logging.info('Done')