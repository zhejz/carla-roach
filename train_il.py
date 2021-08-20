import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
from pathlib import Path
import subprocess
import numpy as np
from stable_baselines3.common.utils import set_random_seed

from carla_gym.utils import config_utils

log = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='train_il')
def main(cfg: DictConfig):
    set_random_seed(cfg.seed, using_cuda=False)

    # caching dataset on the node
    # make sure the first one is the bc (behavior cloning) dataset
    bc_dataset_path = cfg.dagger_datasets[-1]
    if os.path.isdir(bc_dataset_path):
        log.info(f'Copying from {bc_dataset_path} to {cfg.cache_dir}')
        subprocess.call(f'rsync -a --info=progress2 {bc_dataset_path} {cfg.cache_dir}', shell=True)
    else:
        try:
            api = wandb.Api()
            run = api.run(bc_dataset_path)
            log.info(f'Downloading dataset from wandb run: {bc_dataset_path}')
            all_hf = [f for f in run.files() if '.h5' in f.name]
            for i, data_hf in enumerate(all_hf):
                log.info(f'{i+1}/{len(all_hf)}: Downloading {data_hf.name} to {cfg.cache_dir}')
                data_hf.download(replace=True, root=cfg.cache_dir)
        except:
            log.warning(f'Error downloading dataset from wandb run {bc_dataset_path}')
    list_bc_h5 = list(Path(cfg.cache_dir).glob('expert/*.h5'))
    n_ep_bc = len(list_bc_h5)
    bc_size = sum(f.stat().st_size for f in list_bc_h5)
    log.info(f'BC dataset {bc_dataset_path} size: {bc_size/1024**3:.2f}G')

    # downloading dagger dataset and replace bc dataset
    for path in cfg.dagger_datasets[:-1]:
        delete_size = 0
        if len(list_bc_h5) < 0.2*n_ep_bc:
            log.warning(f'Not enough BC episode left ({len(list_bc_h5)}), discard dagger dataset {path}.')
            # break for loop
            break
        else:
            # pre delete 10% to save disk space
            n_episode_pre_delete = int(min(n_ep_bc*0.1, len(list_bc_h5)-0.2*n_ep_bc))
            for _ in range(n_episode_pre_delete):
                f = list_bc_h5.pop(np.random.choice(len(list_bc_h5)))
                delete_size += f.stat().st_size
                log.info(f'Delete {f.name}')
                f.unlink()
        # download dagger dataset
        dagger_size = 0
        if os.path.isdir(path):
            log.info(f'Copying from {path} to {cfg.cache_dir}')
            subprocess.call(f'rsync -a --info=progress2 {path} {cfg.cache_dir}', shell=True)
            dagger_size = sum(f.stat().st_size for f in Path(path).glob('*.h5'))
        else:
            try:
                api = wandb.Api()
                run = api.run(path)
                log.info(f'Downloading dataset from wandb run: {path}')
                all_hf = [f for f in run.files() if '.h5' in f.name]
                for i, data_hf in enumerate(all_hf):
                    log.info(f'{i+1}/{len(all_hf)}: Downloading {data_hf.name} to {cfg.cache_dir}')
                    data_hf.download(replace=True, root=cfg.cache_dir)
                    dagger_size += data_hf.size
            except:
                log.warning(f'Error downloading dataset from wandb run {path}')
        # delete the rest
        while delete_size < dagger_size:
            if len(list_bc_h5) < 0.2*n_ep_bc:
                log.warning(f'Not enough BC episode left ({len(list_bc_h5)}), stop deleting expert dataset.')
                # break while loop
                break
            else:
                f = list_bc_h5.pop(np.random.choice(len(list_bc_h5)))
                delete_size += f.stat().st_size
                log.info(f'Delete {f.name}')
                f.unlink()
        bc_size = sum(f.stat().st_size for f in list_bc_h5)
        log.info(f'BC dataset: {len(list_bc_h5)} episodes, {bc_size/1024**3:.2f}G. '
                 f'Dagger dataset {path}: {dagger_size/1024**3:.2f}G')

    log.info(f"train_il.py working directory: {os.getcwd()}")

    assert len(cfg.agent) == 1, 'Only one agent can be trained at one time.'

    agent_name = next(iter(cfg.agent))
    cfg_agent = cfg.agent[agent_name]
    OmegaConf.save(config=cfg_agent, f='config_agent.yaml')

    AgentClass = config_utils.load_entry_point(cfg_agent.entry_point)
    agent = AgentClass('config_agent.yaml')

    # init wandb: save config_agent
    wandb.init(project=cfg.wb_project, name=cfg.wb_name, group=cfg.wb_group, notes=cfg.wb_notes, tags=cfg.wb_tags)
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['working_dir'] = os.getcwd()
    wandb.config.update(cfg_dict)
    wandb.save('.hydra/*')
    wandb.save('config_agent.yaml')

    agent.learn(cfg.cache_dir, cfg.train_epochs, cfg.reset_step)


if __name__ == '__main__':
    main()
    log.info("train_il.py DONE!")
