import logging


from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor, Normalize


def get_train_transform(cfg):
    trfm = Compose([RandomHorizontalFlip(), ToTensor(),
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return trfm

def get_val_transform(cfg):
    trfm = Compose([ToTensor(),
                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return trfm


def get_loader(cfg, split):
    if split == 'train':
        img_dir = Path(cfg.dataset_dir) / 'train'
    elif split == 'val':
        img_dir = Path(cfg.dataset_dir) / 'test'
    else:
        raise ValueError('Unexpected value for split {split}')

    class_dirs = sorted(img_dir.iterdir())
    class_counts = {p.stem: len(list(p.glob('*.jpg'))) for p in class_dirs}
    # labels = {name.stem: num for num, name in enumerate(class_dirs)}
    
    logging.info(f'Loading {split} data from {img_dir}')
    assert len(class_dirs) == cfg.n_classes, \
        f"Number of classes in cfg and in {split} data folder don't match"
    
    if split == 'train':
        trfm = get_train_transform(cfg)
    else:
        trfm = get_val_transform(cfg)
    
    dataset = ImageFolder(root=img_dir, transform=trfm)

    shuffle = (split == 'train')
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                        shuffle=shuffle, num_workers=0)
    logging.info(f'Existing labels:\n{dataset.class_to_idx}')
    idx_to_class = {idx: cls_name for cls_name, idx in dataset.class_to_idx.items()}
    
    class_counts = [class_counts[idx_to_class[v]]
                            for v in sorted(idx_to_class.keys())]
    logging.info(f'Class counts:\n{class_counts}')
    return loader, dataset, class_counts