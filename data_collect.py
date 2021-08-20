import gym
import math
from pathlib import Path
import json
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import subprocess
import os
import sys

from stable_baselines3.common.vec_env.base_vec_env import tile_images
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from carla_gym.utils import config_utils
from carla_gym.utils.expert_noiser import ExpertNoiser
from utils import saving_utils, server_utils
from agents.rl_birdview.utils.wandb_callback import WandbCallback

log = logging.getLogger(__name__)


def collect_single(run_name, env, data_writer, driver_dict, driver_log_dir, coach_dict, coach_log_dir,
                   dagger_thresholds, log_video, noise_lon=False, noise_lat=False, alpha_coach=None,
                   remove_final_steps=True):

    list_debug_render = []
    list_data_render = []
    ep_stat_dict = {}
    ep_event_dict = {}
    for actor_id, driver in driver_dict.items():
        log_dir = driver_log_dir / actor_id
        log_dir.mkdir(parents=True, exist_ok=True)
        driver.reset(log_dir / f'{run_name}.log')

    for actor_id, coach in coach_dict.items():
        log_dir = coach_log_dir / actor_id
        log_dir.mkdir(parents=True, exist_ok=True)
        coach.reset(log_dir / f'{run_name}.log')

    longitudinal_noiser = ExpertNoiser('Throttle', frequency=15, intensity=10, min_noise_time_amount=2.0) \
        if noise_lon else None
    lateral_noiser = ExpertNoiser('Spike', frequency=25, intensity=4, min_noise_time_amount=0.5) \
        if noise_lat else None

    obs = env.reset()
    timestamp = env.timestamp
    done = {'__all__': False}
    valid = True
    while not done['__all__']:
        driver_control = {}
        coach_control = {}
        driver_supervision = {}
        coach_supervision = {}

        for actor_id, driver in driver_dict.items():
            driver_control[actor_id] = driver.run_step(obs[actor_id], timestamp)
            driver_supervision[actor_id] = driver.supervision_dict

            if noise_lon:
                driver_control[actor_id], _, _ = longitudinal_noiser.compute_noise(
                    driver_control[actor_id], obs[actor_id]['speed']['forward_speed'][0] * 3.6)
            if noise_lat:
                driver_control[actor_id], _, _ = lateral_noiser.compute_noise(
                    driver_control[actor_id], obs[actor_id]['speed']['forward_speed'][0] * 3.6)

        for actor_id, coach in coach_dict.items():
            coach_control[actor_id] = coach.run_step(obs[actor_id], timestamp)
            coach_supervision[actor_id] = coach.supervision_dict

            if alpha_coach:
                if np.random.uniform(0, 1) > alpha_coach:
                    # execute driver
                    c_driver = driver_control[actor_id]
                    coach_supervision[actor_id]['action'] = np.array(
                        [c_driver.throttle, c_driver.steer, c_driver.brake], dtype=np.float32),
                else:
                    # execute coach
                    driver_control[actor_id] = coach_control[actor_id]

        new_obs, reward, done, info = env.step(driver_control)

        if coach_supervision == {}:
            im_rgb = data_writer.write(timestamp=timestamp, obs=obs,
                                       supervision=driver_supervision, reward=reward, control_diff=None)
        else:
            control_diff = {}
            for actor_id in coach_control.keys():
                c_coach = coach_control[actor_id]
                c_driver = driver_control[actor_id]
                control_diff[actor_id] = np.abs(np.array([c_coach.throttle-c_driver.throttle,
                                                          c_coach.steer-c_driver.steer,
                                                          c_coach.brake-c_driver.brake], dtype=np.float32))
            im_rgb = data_writer.write(timestamp=timestamp, obs=obs,
                                       supervision=coach_supervision, reward=reward, control_diff=control_diff)
        obs = new_obs

        debug_imgs = []
        for actor_id, driver in driver_dict.items():
            if log_video:
                if actor_id in coach_supervision:
                    action_str = np.array2string(coach_supervision[actor_id]['action'],
                                                 precision=2, separator=',', suppress_small=True)
                    diff_str = np.array2string(control_diff[actor_id],
                                                 precision=2, separator=',', suppress_small=True)
                    info[actor_id]['terminal_debug']['debug_texts'].append(f'coach_a{action_str}')
                    info[actor_id]['terminal_debug']['debug_texts'].append(f'ct_diff{diff_str}')

                debug_imgs.append(driver.render(info[actor_id]['reward_debug'], info[actor_id]['terminal_debug']))
            if done[actor_id] and (actor_id not in ep_stat_dict):
                episode_stat = info[actor_id]['episode_stat']
                ep_stat_dict[actor_id] = episode_stat
                ep_event_dict[actor_id] = info[actor_id]['episode_event']

                if alpha_coach:
                    if actor_id in coach_dict:
                        _ = coach_dict[actor_id].run_step(obs[actor_id], timestamp)
                        last_value = coach_dict[actor_id].supervision_dict['value']
                    else:
                        _ = driver.run_step(obs[actor_id], timestamp)
                        last_value = driver.supervision_dict['value']
                    valid = data_writer.close(
                        info[actor_id]['terminal_debug'],
                        dagger_thresholds, remove_final_steps, last_value)
                else:
                    valid = data_writer.close(
                        info[actor_id]['terminal_debug'],
                        dagger_thresholds, remove_final_steps, None)
                log.info(f'Episode {run_name} done, valid={valid}')

        if log_video:
            list_debug_render.append(tile_images(debug_imgs))
            list_data_render.append(im_rgb)
        timestamp = env.timestamp

    return valid, list_debug_render, list_data_render, ep_stat_dict, ep_event_dict, timestamp


@ hydra.main(config_path='config', config_name='data_collect')
def main(cfg: DictConfig):
    if cfg.host == 'localhost' and cfg.kill_running:
        server_utils.kill_carla()
    log.setLevel(getattr(logging, cfg.log_level.upper()))

    # start carla servers
    server_manager = server_utils.CarlaServerManager(cfg.carla_sh_path, port=cfg.port)
    server_manager.start()

    # single actor, place holder for multi actors
    driver_dict = {}
    coach_dict = {}
    obs_configs = {}
    reward_configs = {}
    terminal_configs = {}
    for ev_id, ev_cfg in cfg.actors.items():
        # initiate driver agent
        cfg_driver = cfg.agent[ev_cfg.driver]
        OmegaConf.save(config=cfg_driver, f='config_driver.yaml')
        DriverAgentClass = config_utils.load_entry_point(cfg_driver.entry_point)
        driver_dict[ev_id] = DriverAgentClass('config_driver.yaml')
        obs_configs[ev_id] = driver_dict[ev_id].obs_configs

        # initiate coach agent, expert obs always override cilrs obs
        if ev_cfg.coach is None:
            # no coach, first round data collection, add cilrs obs if missing
            for k, v in OmegaConf.to_container(cfg.agent.cilrs.obs_configs).items():
                if k not in obs_configs[ev_id]:
                    obs_configs[ev_id][k] = v
        else:
            # dagger: cilrs driving, given coach, obs_configs from expert override cilrs config
            cfg_coach = cfg.agent[ev_cfg.coach]
            OmegaConf.save(config=cfg_coach, f='config_coach.yaml')
            CoachAgentClass = config_utils.load_entry_point(cfg_coach.entry_point)
            coach_dict[ev_id] = CoachAgentClass('config_coach.yaml')
            for k, v in coach_dict[ev_id].obs_configs.items():
                obs_configs[ev_id][k] = v

        # get obs_configs from agent
        reward_configs[ev_id] = OmegaConf.to_container(ev_cfg.reward)
        terminal_configs[ev_id] = OmegaConf.to_container(ev_cfg.terminal)

    # check h5 birdview maps have been generated
    config_utils.check_h5_maps(cfg.test_suites, obs_configs, cfg.carla_sh_path)

    # resume env_idx from checkpoint.txt
    last_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/checkpoint.txt'
    if cfg.resume and os.path.isfile(last_checkpoint_path):
        with open(last_checkpoint_path, 'r') as f:
            env_idx = int(f.read())
    else:
        env_idx = 0

    # resume task_idx from ep_stat_buffer_{env_idx}.json
    ep_state_buffer_json = f'{hydra.utils.get_original_cwd()}/outputs/ep_stat_buffer_{env_idx}.json'
    if cfg.resume and os.path.isfile(ep_state_buffer_json):
        ep_stat_buffer = json.load(open(ep_state_buffer_json, 'r'))
        ckpt_task_idx = len(ep_stat_buffer['hero'])
    else:
        ckpt_task_idx = 0
        ep_stat_buffer = {}
        for actor_id in driver_dict.keys():
            ep_stat_buffer[actor_id] = []

    # resume wandb run
    wb_checkpoint_path = f'{hydra.utils.get_original_cwd()}/outputs/wb_run_id.txt'
    if cfg.resume and os.path.isfile(wb_checkpoint_path):
        with open(wb_checkpoint_path, 'r') as f:
            wb_run_id = f.read()
    else:
        wb_run_id = None

    log.info(f'Start from env_idx: {env_idx}, task_idx {ckpt_task_idx}')

    # make directories
    dataset_root = Path(cfg.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    im_stack_idx = [-1]
    if cfg.actors.hero.driver == 'cilrs':
        ckpt_project = cfg.agent.cilrs.wb_run_path.split('/')[1]
        ckpt_run_id = cfg.agent.cilrs.wb_run_path.split('/')[2]
        dataset_name = f'{ckpt_project}/{ckpt_run_id}'
        cfg.wb_project = ckpt_project
        wb_run_name = f'{dataset_root.name}/{dataset_name}'
        im_stack_idx = driver_dict[cfg.ev_id]._env_wrapper.im_stack_idx
    else:
        dataset_name = 'expert'
        wb_run_name = f'{dataset_root.name}/{dataset_name}'

    dataset_dir = Path(cfg.dataset_root) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    diags_dir = Path('diagnostics')
    driver_log_dir = Path('driver_log')
    coach_log_dir = Path('coach_log')
    video_dir = Path('videos')
    diags_dir.mkdir(parents=True, exist_ok=True)
    driver_log_dir.mkdir(parents=True, exist_ok=True)
    coach_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # init wandb
    wandb.init(project=cfg.wb_project, name=wb_run_name, group=cfg.wb_group, notes=cfg.wb_notes, tags=cfg.wb_tags,
               id=wb_run_id, resume="allow")
    wandb.config.update(OmegaConf.to_container(cfg))
    wandb.save('./config_agent.yaml')
    with open(wb_checkpoint_path, 'w') as f:
        f.write(wandb.run.id)

    if env_idx >= len(cfg.test_suites):
        log.info(f'Finished! env_idx: {env_idx}, resave to wandb')
        if cfg.save_to_wandb:
            wandb.save(f'{dataset_dir.as_posix()}/*.h5', base_path=cfg.dataset_root)
        return

    # make env
    env_setup = OmegaConf.to_container(cfg.test_suites[env_idx])
    env = gym.make(env_setup['env_id'], obs_configs=obs_configs, reward_configs=reward_configs,
                   terminal_configs=terminal_configs, host=cfg.host, port=cfg.port,
                   seed=cfg.seed, no_rendering=cfg.no_rendering, **env_setup['env_configs'])

    # main loop
    n_episodes_per_env = math.ceil(cfg.n_episodes/len(cfg.test_suites))

    for task_idx in range(ckpt_task_idx, n_episodes_per_env):
        idx_episode = task_idx + n_episodes_per_env * env_idx
        run_name = f'{idx_episode:04}'

        while True:
            env.set_task_idx(np.random.choice(env.num_tasks))

            data_writer = saving_utils.DataWriter(dataset_dir/f'{run_name}.h5', cfg.ev_id, im_stack_idx)

            noise_lon = cfg.inject_noise and np.random.randint(101) < 20
            noise_lat = cfg.inject_noise and np.random.randint(101) < 20

            log.info(f'Start episode {run_name}, noise_lon={noise_lon}, noise_lat={noise_lat}, {env_setup}')
            valid, list_debug_render, list_data_render, ep_stat_dict, ep_event_dict, timestamp = collect_single(
                run_name, env, data_writer, driver_dict, driver_log_dir,
                coach_dict, coach_log_dir, cfg.dagger_thresholds, cfg.log_video, noise_lon, noise_lat,
                cfg.alpha_coach, cfg.remove_final_steps)
            if valid:
                break

        # log videos
        if cfg.log_video:
            debug_video_path = (video_dir / f'debug_{run_name}.mp4').as_posix()
            encoder = ImageEncoder(debug_video_path, list_debug_render[0].shape, 30, 30)
            for im in list_debug_render:
                encoder.capture_frame(im)
            encoder.close()
            wandb.log({f'video/debug_{run_name}': wandb.Video(debug_video_path)}, step=idx_episode)
            if cfg.actors.hero.driver != 'cilrs':
                data_video_path = (video_dir / f'data_{run_name}.mp4').as_posix()
                encoder = ImageEncoder(data_video_path, list_data_render[0].shape, 30, 30)
                for im in list_data_render:
                    encoder.capture_frame(im)
                encoder.close()
                wandb.log({f'video/data_{run_name}': wandb.Video(data_video_path)}, step=idx_episode)
            encoder = None

        # dump events
        diags_json_path = (diags_dir / f'{run_name}.json').as_posix()
        with open(diags_json_path, 'w') as fd:
            json.dump(ep_event_dict, fd, indent=4, sort_keys=False)

        # save diags and agents_log
        wandb.save(diags_json_path)
        wandb.save(f'{driver_log_dir.as_posix()}/*/*')
        wandb.save(f'{coach_log_dir.as_posix()}/*/*')

        # save time
        wandb.log({'time/total_step': timestamp['step'],
                   'time/fps':  timestamp['step'] / timestamp['relative_wall_time']
                   }, step=idx_episode)

        # save statistics
        for actor_id, ep_stat in ep_stat_dict.items():
            ep_stat_buffer[actor_id].append(ep_stat)
            log_dict = {}
            for k, v in ep_stat.items():
                k_actor = f'{actor_id}/{k}'
                log_dict[k_actor] = v
            wandb.log(log_dict, step=idx_episode)

        with open(ep_state_buffer_json, 'w') as fd:
            json.dump(ep_stat_buffer, fd, indent=4, sort_keys=True)
        # clean up
        list_debug_render.clear()
        list_data_render.clear()
        ep_stat_dict = None
        ep_event_dict = None

        saving_utils.report_dataset_size(dataset_dir)
        dataset_size = subprocess.check_output(['du', '-sh', dataset_dir]).split()[0].decode('utf-8')
        log.warning(f'{dataset_dir}: dataset_size {dataset_size}')

    # close env
    env.close()
    env = None
    server_manager.stop()

    # log after all episodes are completed
    table_data = []
    ep_stat_keys = None
    for actor_id, list_ep_stat in json.load(open(ep_state_buffer_json, 'r')).items():
        avg_ep_stat = WandbCallback.get_avg_ep_stat(list_ep_stat)
        data = [actor_id, cfg.actors[actor_id].driver, env_idx, str(len(list_ep_stat))]
        if ep_stat_keys is None:
            ep_stat_keys = list(avg_ep_stat.keys())
        data += [f'{avg_ep_stat[k]:.4f}' for k in ep_stat_keys]
        table_data.append(data)

    table_columns = ['actor_id', 'driver', 'env_idx',  'n_episode'] + ep_stat_keys
    wandb.log({'table/summary': wandb.Table(data=table_data, columns=table_columns)})

    with open(last_checkpoint_path, 'w') as f:
        f.write(f'{env_idx+1}')

    log.info(f"Finished data collection env_idx {env_idx}, {env_setup['env_id']}.")
    if env_idx+1 == len(cfg.test_suites):
        if cfg.save_to_wandb:
            wandb.save(f'{dataset_dir.as_posix()}/*.h5', base_path=cfg.dataset_root)
        log.info(f"Finished, {env_idx+1}/{len(cfg.test_suites)}")
        return
    else:
        log.info(f"Not finished, {env_idx+1}/{len(cfg.test_suites)}")
        sys.exit(1)

    return


if __name__ == '__main__':
    main()
    log.info("data_collect.py DONE!")
