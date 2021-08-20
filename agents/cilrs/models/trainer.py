import logging
import torch as th
import torch.optim as optim
import torch.nn as nn
import time
import wandb
from pathlib import Path
import numpy as np
import copy

from .utils.dataset import get_dataloader
from .utils.branched_loss import BranchedLoss

log = logging.getLogger(__name__)


class Trainer():
    def __init__(self, policy,
                 batch_size=64,
                 num_workers=3,
                 learning_rate=0.0002,
                 lr_schedule_factor=0.1,
                 branch_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 action_weights=[0.5, 0.5],
                 speed_weight=0.05,
                 value_weight=0.0,
                 features_weight=0.0,
                 l1_loss=True,
                 action_kl=True,
                 action_agg=None,
                 action_mll=None,
                 starting_iteration=0,
                 starting_epoch=0,
                 im_augmentation=None
                 ):

        self._init_kwargs = copy.deepcopy(locals())
        del self._init_kwargs['self']
        del self._init_kwargs['policy']

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # multi-gpu
        self.num_gpus = th.cuda.device_count()
        log.info(f'number of gpus: {self.num_gpus}')

        # kwargs for dataloader
        self.batch_size = batch_size * self.num_gpus
        self.num_workers = num_workers
        self.im_augmentation = im_augmentation
        self.lr_schedule_factor = lr_schedule_factor

        # kwargs that are changing
        self.starting_iteration = starting_iteration
        self.starting_epoch = starting_epoch

        self.iteration = starting_iteration

        self.criterion = BranchedLoss(branch_weights, action_weights, speed_weight,
                                      value_weight, features_weight, l1_loss, action_kl, action_agg, action_mll)

        self.policy = policy.to(self.device)
        # print number of model params
        model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        total_params = sum([np.prod(p.size()) for p in model_parameters])
        log.info(f'trainable parameters: {total_params/1000000:.2f}M')

        # optimizer / lr_scheduler
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = self.get_lr_scheduler()

        # path to save ckpt
        self._ckpt_dir = Path('ckpt')
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_lr_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', min_lr=1e-7,
                                                    factor=self.lr_schedule_factor,
                                                    patience=5)

    def learn(self, dataset_dir, train_epochs, env_wrapper, reset_step=False):
        if reset_step:
            self.starting_iteration = 0
            self.starting_epoch = 0
            self.iteration = 0
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            self.scheduler = self.get_lr_scheduler()

        train_dataset, val_dataset = get_dataloader(dataset_dir, env_wrapper,
                                                    self.im_augmentation, self.batch_size, self.num_workers)
        # multi-gpu
        if self.num_gpus > 1:
            self.policy = nn.DataParallel(self.policy)

        t0 = time.time()
        log.info('Start Training')

        best_val_loss = 1e10
        best_val_loss_epoch = self.starting_epoch
        for idx_epoch in range(self.starting_epoch, train_epochs):
            # train
            self._train(train_dataset)

            # val
            t_val = time.time()
            val_loss, val_action_loss, val_speed_loss, val_value_loss, val_features_loss = \
                self._validate(val_dataset, idx_epoch)
            wandb.log({
                'val/loss': val_loss,
                'val/action_loss': val_action_loss,
                'val/speed_loss': val_speed_loss,
                'val/value_loss': val_value_loss,
                'val/features_loss': val_features_loss,
                'time/val_time': time.time()-t_val
            }, step=self.iteration)
            wandb.log({'train/lr': self.optimizer.param_groups[0]['lr']}, step=self.iteration)

            # update lr
            self.scheduler.step(val_loss)

            # save checkpoint
            # if (idx_epoch==0) or (idx_epoch>5 and val_loss < best_val_loss):
            # if (idx_epoch >= 25) and ((val_loss < best_val_loss) or (idx_epoch % 5 == 0)):
            #     self.starting_epoch = idx_epoch+1
            #     self.starting_iteration = self.iteration+1
            #     ckpt_path = (self._ckpt_dir / f'ckpt_{idx_epoch}.pth').as_posix()
            #     self.save(ckpt_path)
            #     log.info(f'Save ckpt, val_loss: {val_loss:.6f} path: {ckpt_path}')
            #     wandb.save(ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = idx_epoch

            log.info(f'Epoch {idx_epoch}: {(time.time()-t0)/3600:.2f} hours, '
                     f'best_val_loss: {best_val_loss:.6f}, best_val_loss_epoch: {best_val_loss_epoch}')
            self.starting_epoch = idx_epoch+1
            self.starting_iteration = self.iteration+1

        ckpt_path = (self._ckpt_dir / f'ckpt_{idx_epoch}.pth').as_posix()
        self.save(ckpt_path)
        log.info(f'Save ckpt, val_loss: {val_loss:.6f} path: {ckpt_path}')
        wandb.save(ckpt_path)
        log.info('Learn Finished')

    def _train(self, dataset):
        self.policy = self.policy.train()

        for command, policy_input, supervision in dataset:
            t0 = time.time()
            # mem_available = psutil.virtual_memory().available
            # print(f'memory available {mem_available/1e9:.2f}GB')

            policy_input = dict([(k, th.as_tensor(v).to(self.device)) for k, v in policy_input.items()])
            supervision = dict([(k, th.as_tensor(v).to(self.device)) for k, v in supervision.items()])
            command = th.as_tensor(command).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.policy.forward(**policy_input)

            action_loss, speed_loss, value_loss, features_loss = self.criterion.forward(outputs, supervision, command)
            loss = action_loss+speed_loss+value_loss+features_loss
            loss.backward()
            self.optimizer.step()

            wandb.log({
                'train/loss': loss.item(),
                'train/action_loss': action_loss.item(),
                'train/speed_loss': speed_loss.item(),
                'train/value_loss': value_loss.item(),
                'train/features_loss': features_loss.item(),
                'time/train_fps': self.batch_size / (time.time()-t0)
            }, step=self.iteration)
            self.iteration += self.batch_size

    def _validate(self, dataset, idx_epoch):
        self.policy = self.policy.eval()

        losses = []
        action_losses = []
        speed_losses = []
        value_losses = []
        features_losses = []
        for command, policy_input, supervision in dataset:

            policy_input = dict([(k, th.as_tensor(v).to(self.device)) for k, v in policy_input.items()])
            supervision = dict([(k, th.as_tensor(v).to(self.device)) for k, v in supervision.items()])
            command = th.as_tensor(command).to(self.device)

            # controls = data['cmd']
            with th.no_grad():
                outputs = self.policy.forward(**policy_input)
                action_loss, speed_loss, value_loss, features_loss = self.criterion.forward(
                    outputs, supervision, command)
                loss = action_loss+speed_loss+value_loss+features_loss
                losses.append(loss.item())
                action_losses.append(action_loss.item())
                speed_losses.append(speed_loss.item())
                value_losses.append(value_loss.item())
                features_losses.append(features_loss.item())

        loss = np.mean(losses)
        action_loss = np.mean(action_losses)
        speed_loss = np.mean(speed_losses)
        value_loss = np.mean(value_losses)
        features_loss = np.mean(features_losses)

        # if idx_epoch == 0:
        #     wandb.log({'inspect/im_val': [wandb.Image(policy_input['im'], caption="val")]}, step=idx_epoch)

        return loss, action_loss, speed_loss, value_loss, features_loss

    def save(self, path: str):
        if self.num_gpus > 1:
            policy_state_dict = self.policy.module.state_dict()
            policy_init_kwargs = self.policy.module.init_kwargs
        else:
            policy_state_dict = self.policy.state_dict()
            policy_init_kwargs = self.policy.init_kwargs

        self._init_kwargs['starting_epoch'] = self.starting_epoch
        self._init_kwargs['starting_iteration'] = self.starting_iteration

        th.save({'policy_state_dict': policy_state_dict,
                 'policy_init_kwargs': policy_init_kwargs,
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict(),
                 'trainer_init_kwargs': self._init_kwargs},
                path)

    @classmethod
    def load(cls, policy, path):
        saved_variables = th.load(path)
        trainer = cls(policy, **saved_variables['trainer_init_kwargs'])
        trainer.optimizer.load_state_dict(saved_variables['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(saved_variables['scheduler_state_dict'])
        return trainer
