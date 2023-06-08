import time
import wandb
from types import SimpleNamespace
from logger.utils import time_since
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


def get_file_logger(filename):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class Logger:

    def __init__(
            self,
            train_steps,
            valid_steps,
            config: SimpleNamespace,
            eval_steps,
            output_file,
    ):

        self.train_steps = train_steps
        self.valid_steps = valid_steps

        self.config = config
        self.train_print_frequency = config.logger.train_print_frequency
        self.valid_print_frequency = config.logger.valid_print_frequency

        self.eval_steps = eval_steps

        self.train_epoch_start_time = None
        self.valid_epoch_start_time = None

        self.train_step = 0
        self.valid_step = 0

        self.logger = get_file_logger(output_file)

    def log(self, text):
        self.logger.info(text)

    def train_epoch_start(self):
        self.train_epoch_start_time = time.time()
        self.train_step = 0

    def valid_epoch_start(self):
        self.valid_epoch_start_time = time.time()
        self.valid_step = 0

    def train_epoch_end(self, epoch, train_losses, valid_losses):
        elapsed = time.time() - self.train_epoch_start_time
        self.log(
            f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.8f} '
            f'avg_val_loss: {valid_losses.avg:.8f} time: {elapsed:.0f}s\n'
            '=============================================================================\n'
        )
        if self.config.logger.use_wandb:
            step = epoch + 1
            wandb.log(
                {
                    'epoch': step,
                    'train_loss': train_losses.avg,
                },
            )

    def valid_epoch_end(self, epoch, valid_losses, score):
        self.log(f'\tEpoch {epoch + 1} - valid_loss: {valid_losses.avg} - Score: {score}')
        if self.config.logger.use_wandb:
            step = (epoch + (self.eval_steps.index(self.train_step) + 1) / (len(self.eval_steps)))
            wandb.log(
                {
                    'epoch': step,
                    'score': score,
                    'valid_loss': valid_losses.avg
                },
            )

    def train_step_end(self, epoch, train_losses, grad_norm, scheduler):

        if (self.train_step % self.train_print_frequency == 0) or \
                (self.train_step == (self.train_steps - 1)) or \
                (self.train_step + 1 in self.eval_steps) or \
                (self.train_step - 1 in self.eval_steps):
            remain = time_since(self.train_epoch_start_time, float(self.train_step + 1) / self.train_steps)

            self.log(
                f'Epoch: [{epoch + 1}][{self.train_step + 1}/{self.train_steps}] '
                f'Elapsed {remain:s} '
                f'Loss: {train_losses.val:.8f}({train_losses.avg:.8f}) '
                f'Grad: {grad_norm:.4f}  '
                f'LR: {scheduler.get_last_lr()[0]:.8f} '
            )

        if self.config.logger.use_wandb:
            wandb.log(
                {
                    'avg_train_loss': train_losses.avg,
                    'lr': scheduler.get_last_lr()[0]
                },
            )

        self.train_step += 1

    def valid_step_end(self, epoch, valid_losses):

        if (self.valid_step % self.valid_print_frequency == 0) or \
                (self.valid_step == (self.valid_steps - 1)):

            remain = time_since(self.valid_epoch_start_time, float(self.valid_step + 1) / self.valid_steps)
            self.log(
                f'\tEVAL: [{epoch + 1}][{self.valid_step + 1}/{self.valid_steps}] '
                f'Elapsed: {remain:s} '
                f'Loss: {valid_losses.avg:.8f} '
            )

        self.valid_step += 1

    def valid_score_improved(self, epoch, score):
        self.log(f'\tEpoch {epoch + 1} - Save Best Score: ({score:.6f})\n')

        if self.config.logger.use_wandb:
            wandb.log({'best_score': score})
