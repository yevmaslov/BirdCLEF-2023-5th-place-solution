import torch
import torch.nn as nn
from types import SimpleNamespace
import os
import numpy as np
import gc
import sys
sys.path.append('../')
sys.path.append('birdcleff-2023')
from src.logger.utils import AverageMeter, time_since
from src.models.utils import collate_dict, batch_to_device


class Trainer:

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            valid_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            logger,
            config: SimpleNamespace,
            compute_score_fn: callable,
            output_dir,
            eval_steps,
            direction='minimize'
    ) -> None:

        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.logger = logger

        self.compute_score_fn = compute_score_fn

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.training.apex)
        self.eval_steps = eval_steps

        self.direction = direction

        self.epoch = 0
        self.best_score = np.inf if self.direction == 'minimize' else 0

        self.output_dir = output_dir
        self.checkpoint_path = self.output_dir / 'chkp' / f'fold_{self.config.fold}_chkp.pth'
        self.best_model_path = self.output_dir / 'models' / f'fold_{self.config.fold}_best.pth'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def validate(self):
        valid_losses = AverageMeter()
        self.model.eval()
        self.logger.valid_epoch_start()

        predictions = []

        for step, inputs in enumerate(self.valid_dataloader):
            inputs = batch_to_device(inputs, self.device)

            with torch.no_grad():
                y_pred, loss = self.model(inputs)

            predictions.append(y_pred.detach().to('cpu').numpy())

            valid_losses.update(loss.item(), self.config.dataset.valid_batch_size)
            self.logger.valid_step_end(self.epoch, valid_losses)

        predictions = np.concatenate(predictions)
        return valid_losses, predictions

    def train(self):
        if os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()

        for epoch in range(self.epoch, self.config.training.epochs):
            self.model.train()
            # if self.config.model.finetune_head:
            #     self.model.model.eval()
                
            if self.config.model.finetune_top_layers:
                self.model.model.blocks[:self.config.freeze_n_layers].eval()
                self.model.model.conv_stem.eval()
                self.model.model.bn1.eval()
                # self.model.eval()

            train_losses = AverageMeter()
            valid_losses = AverageMeter()
            self.logger.train_epoch_start()

            for step, inputs in enumerate(self.train_dataloader):
                inputs = collate_dict(inputs)
                inputs = batch_to_device(inputs, self.device)

                batch_size = self.config.dataset.train_batch_size
                if self.config.training.apex:
                    with torch.cuda.amp.autocast():
                        y_pred, loss = self.model(inputs)
                else:
                    y_pred, loss = self.model(inputs)

                train_losses.update(loss.item(), batch_size)

                if self.config.training.apex:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                               self.config.training.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                self.logger.train_step_end(self.epoch, train_losses, grad_norm, self.scheduler)

                if (step + 1) in self.eval_steps:
                    valid_losses, predictions = self.validate()
                    score = self.compute_score_fn(self.model, valid_losses, predictions)
                    self.logger.valid_epoch_end(self.epoch, valid_losses, score)

                    self.model.train()
                    
                    # if self.config.model.finetune_head:
                    #     self.model.model.eval()
                        
                    if self.config.model.finetune_top_layers:
                        self.model.model.blocks[:self.config.freeze_n_layers].eval()
                        self.model.model.conv_stem.eval()
                        self.model.model.bn1.eval()
                        # self.model.eval()

                    if (self.direction == 'minimize' and score <= self.best_score) or \
                            (self.direction == 'maximize' and score >= self.best_score):
                        self.best_score = score
                        self.save_best_model()
                        self.logger.valid_score_improved(epoch, score)

            self.save_checkpoint()
            self.logger.train_epoch_end(self.epoch, train_losses, valid_losses)
            self.logger.log(f'BN Running mean: {self.model.model.bn1.running_mean.sum()}')
            self.logger.log(f'BN Running mean: {self.model.model.blocks[0][0].bn1.running_mean.sum()}')
            self.logger.log(f'BN Running mean: {self.model.model.blocks[1][0].bn1.running_mean.sum()}')
            self.logger.log(f'BN Running mean: {self.model.model.blocks[2][0].bn1.running_mean.sum()}')
            self.logger.log(f'BN Running mean: {self.model.model.blocks[3][0].bn1.running_mean.sum()}')
            self.logger.log(f'BN Running mean: {self.model.model.blocks[4][0].bn1.running_mean.sum()}\n')
            self.epoch += 1

        torch.cuda.empty_cache()
        gc.collect()
        return None

    def save_best_model(self):
        torch.save(
            {
                'model': self.model.state_dict()
            },
            self.best_model_path
        )

    def save_checkpoint(self):
        torch.save(
            {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'best_score': self.best_score
            },
            self.checkpoint_path
        )

    def load_checkpoint(self):
        self.logger.log('='*20 + ' Checkpoint loaded ' + '='*20)

        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_score = checkpoint['best_score']
