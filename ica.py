import logging
from pathlib import Path

import numpy as np
import torch

from torch import nn

from opt import initalize_schedulers, initialize_optimizers
from utils.visualization import visualize_model_outputs

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class ICAModel(nn.Module):
    
    def __init__(self, encoder, decoder, discriminator, cfg, class_weights=None) -> None:
        super(ICAModel, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.beta = cfg.beta
        
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        
        if class_weights is None:
            self.register_buffer('class_weights', torch.ones(cfg.n_classes) / cfg.n_classes)
        else:
            self.register_buffer('class_weights', torch.tensor(class_weights))

        self.configure_optimizers()

        # setup logging
        log_path = Path(cfg.save_dir) / 'train'
        if log_path.exists():
            experiment_num = len(list(log_path.iterdir())) + 1
        else:
            experiment_num = 1
        self.log_path = log_path / f'exp{experiment_num}'
        self.log_path.mkdir(parents=True, exist_ok=True)
        config_path = self.log_path / 'config.yml'
        with open(config_path, 'w') as f:
            f.write(cfg.dump())
        logging.info(f'Saved config to {config_path}')

        (self.log_path / 'weights').mkdir()
        
        self.recon_loss_func = nn.L1Loss()
        self.disc_loss_func = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        self.ae_optim, self.disc_optim = initialize_optimizers(
                                                    self.encoder, 
                                                    self.decoder,
                                                    self.discriminator,
                                                    self.cfg)

        self.ae_sched, self.disc_sched = initalize_schedulers(
                                                    self.ae_optim,
                                                    self.disc_optim,
                                                    self.cfg)

    def forward(self, x, cond_in, cond_out=None):
        if cond_out is None:
            cond_out = cond_in
        code = self.encoder(x, cond_in)
        logits = self.discriminator(code)
        recon = self.decoder(code, cond_out)
        return recon, code, logits

    def train(self, train_loader, val_loader, n_epochs, vis_dataset=None):
        eps = 1e-6
        best_val_recon_loss = float('inf')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)

        for epoch in range(n_epochs):
            self.encoder.train() 
            self.decoder.train() 
            self.discriminator.train()
            recon_losses = []
            indep_losses = []
            ae_losses = []

            for batch_idx, batch in enumerate(train_loader):
                tr_recon_loss, tr_indep_loss, ae_loss = self.training_step(batch, batch_idx)
                
                recon_losses.append(tr_recon_loss)
                indep_losses.append(tr_indep_loss)
                ae_losses.append(ae_loss)

                if batch_idx % 10 == 0:
                    logging.info(f'Train epoch: {epoch+1}/{n_epochs}, Step: {batch_idx}')
                    logging.info(f'Recon_loss: {np.mean(recon_losses):.3f}')
                    logging.info(f'Indep loss:  {np.mean(indep_losses):.3f}')
                    logging.info(f'AE loss:  {np.mean(ae_losses):.3f}')
                    logging.info(f'Learning rate = {self.ae_optim.param_groups[0]["lr"]:.7f}' )
            
            logging.info(f'Validation epoch: {epoch+1}/{n_epochs}')
            val_recon_loss, val_indep_loss, val_ae_loss, val_indep_gap = self.test(val_loader)

            if (val_recon_loss < best_val_recon_loss - eps) or (epoch + 1 == n_epochs):
                if val_recon_loss < best_val_recon_loss - eps:
                    cpt_path = self.log_path / 'best.pt'
                    best_val_recon_loss = val_recon_loss
                else:
                    cpt_path = self.log_path / 'last.pt'
                self.save_checkpoint(epoch, val_recon_loss,
                                     val_indep_loss, val_indep_gap,
                                     path=cpt_path)
                logging.info(f'Saved checkpoint to {cpt_path}')

            self.ae_sched.step()
            self.disc_sched.step()

            if self.cfg.visualization.show and vis_dataset is not None:
                if epoch > 0 and epoch % self.cfg.visualization.epoch_interval:
                    visualize_model_outputs(vis_dataset,
                                            self.encoder, self.decoder,
                                            labels=list(range(self.cfg.n_classes)),
                                            device=self.cfg.device,
                                            n_examples=self.cfg.visualization.n_examples,
                                            save_path=self.log_path / f'samples_{epoch}.png')

    def test(self, dataloader):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.discriminator.to(self.device)

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        recon_losses = []
        indep_losses = []
        ae_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                recon_loss, indep_loss, ae_loss = self.validation_step(batch, batch_idx)
                recon_losses.append(recon_loss)
                indep_losses.append(indep_loss)
                ae_losses.append(ae_loss)

        recon_loss = np.mean(recon_losses)
        indep_loss = np.mean(indep_losses)
        ae_loss = np.mean(ae_losses)
        indep_gap = abs(indep_loss - np.log(1 / max(self.class_weights).item()))

        logging.info(f'Reconstruction Loss : {recon_loss:.3f}')
        logging.info(f'Independence Loss : {indep_loss:.3f}')
        logging.info(f'Independence gap : {indep_gap:.3f}')
        logging.info(f'AE Loss : {ae_loss:.3f}')

        return recon_loss, indep_loss, ae_loss, indep_gap

    def training_step(self, batch, batch_idx):
        x, cond = batch
        x = x.to(self.device)
        cond = cond.to(self.device)

        if batch_idx % self.cfg.discriminator.opt_interval == 0:
            self.discriminator.zero_grad()
            code = self.encoder(x, cond)
            logits = self.discriminator(code)
            discriminator_loss = self.disc_loss_func(logits, cond)
            discriminator_loss.backward()
            self.disc_optim.step()

        if batch_idx % self.cfg.autoencoder.opt_interval == 0:
            self.encoder.zero_grad()
            self.decoder.zero_grad()
            code = self.encoder(x, cond)
            recon = self.decoder(code, cond)
            logits = self.discriminator(code)
            discriminator_loss = self.disc_loss_func(logits, cond)
            recon_loss = self.recon_loss_func(recon, x)
            indep_loss = - discriminator_loss
            ae_loss = recon_loss + self.beta * indep_loss
            ae_loss.backward()
            self.ae_optim.step()

        return recon_loss.item(), indep_loss.item(), ae_loss.item()

    def validation_step(self, batch, batch_idx):
        x, cond = batch
        x = x.to(self.device)
        cond = cond.to(self.device)
        recon, code, logits = self.forward(x, cond)
        discriminator_loss = self.disc_loss_func(logits, cond)

        recon_loss = self.recon_loss_func(recon, x)
        indep_loss = - discriminator_loss
        ae_loss = recon_loss + self.beta * indep_loss

        return recon_loss.item(), indep_loss.item(), ae_loss.item()

    def save_checkpoint(self, epoch,
                        recon_loss, indep_loss, indep_gap,
                        path):
        cpt = {'epoch': epoch,
               'recon_loss': recon_loss,
               'indep_loss': indep_loss,
               'indep_gap': indep_gap}
        cpt['models'] = self.state_dict()
        cpt['ae_optim'] = self.ae_optim.state_dict()
        cpt['disc_optim'] = self.disc_optim.state_dict()
        torch.save(cpt, path)
        logging.info(f'saved checkpoint to {path}')

    def load_checkpoint(self, path):
        cpt = torch.load(path)
        self.load_state_dict(cpt['models'])
        self.ae_optim.load_state_dict(cpt['ae_optim'])
        self.disc_optim.load_state_dict(cpt['disc_optim'])
        logging.info(f'loaded checkpoint from {path}')