import copy
import numpy as np
import h5py
import logging
import cv2
import copy

log = logging.getLogger(__name__)


def report_dataset_size(dataset_dir):
    list_h5_path = list(dataset_dir.glob('*.h5'))

    total_steps = 0
    critical_steps = 0

    for h5_path in list_h5_path:
        try:
            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as hf:
                total_steps += len(hf)
                for _, group_step in hf.items():
                    if group_step.attrs.get('critical', True):
                        critical_steps += 1
        except:
            log.warning(f'Unalbe to open h5 file: {h5_path}')

    log.warning(f'{dataset_dir}: {len(list_h5_path)} episodes, '
                f'{total_steps} saved frames={total_steps/36000:.2f} hours, '
                f'{critical_steps} critical frames={critical_steps/36000:.2f} hours')


class DataWriter():
    def __init__(self, file_path, ev_id, im_stack_idx=[-1]):
        self._file_path = file_path
        self._ev_id = ev_id
        self._im_stack_idx = np.array(im_stack_idx)
        self._data_list = []

    def write(self, timestamp, obs, supervision, reward, control_diff=None):
        obs = copy.deepcopy(obs)
        supervision = copy.deepcopy(supervision)
        assert self._ev_id in obs and self._ev_id in supervision
        obs = obs[self._ev_id]

        data_dict = {
            'step': timestamp['step'],
            'obs': {
                'central_rgb': None,
                'left_rgb': None,
                'right_rgb': None,
                'gnss': None,
                'speed': None
            },
            'supervision': None,
            'control_diff': None,
            'reward': None,
            'critical': True
        }

        # central_rgb
        data_dict['obs']['central_rgb'] = obs['central_rgb']
        # gnss speed
        data_dict['obs']['gnss'] = obs['gnss']
        data_dict['obs']['speed'] = obs['speed']

        # left_rgb & right_rgb
        if 'left_rgb' in obs and 'right_rgb' in obs:
            data_dict['obs']['left_rgb'] = obs['left_rgb']
            data_dict['obs']['right_rgb'] = obs['right_rgb']
            render_rgb = np.concatenate([obs['central_rgb']['data'],
                                         obs['left_rgb']['data'],
                                         obs['right_rgb']['data']], axis=0)
        else:
            render_rgb = obs['central_rgb']['data']
        render_rgb = copy.deepcopy(render_rgb)

        # supervision
        data_dict['supervision'] = supervision[self._ev_id]

        # reward
        data_dict['reward'] = reward[self._ev_id]

        # control_diff
        if control_diff is not None:
            data_dict['control_diff'] = control_diff[self._ev_id]

        self._data_list.append(data_dict)

        # put text
        action_str = np.array2string(supervision[self._ev_id]['action'],
                                     precision=2, separator=',', suppress_small=True)
        speed = supervision[self._ev_id]['speed']
        txt_1 = f'{action_str} spd:{speed[0]:5.2f}'
        render_rgb = cv2.putText(render_rgb, txt_1, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return render_rgb

    @staticmethod
    def _write_dict_to_group(group, key, my_dict):
        group_key = group.create_group(key)
        for k, v in my_dict.items():
            if type(v) == np.ndarray and v.size > 2000:
                group_key.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            else:
                group_key.create_dataset(k, data=v)

    def close(self, terminal_debug, dagger_thresholds, remove_final_steps, last_value=None):
        # clean up data
        log.info(f'Episode finished, len={len(self._data_list)}')

        if self._data_list[0]['control_diff'] is not None:
            # dagger critical sampling, control_diff=[throttle, steer, brake]
            if remove_final_steps:
                if terminal_debug['traffic_rule_violated']:
                    step_to_delete = min(50, len(self._data_list))
                    del self._data_list[-step_to_delete:]
                    log.warning(f'traffic_rule_violated, len={len(self._data_list)}')

                if terminal_debug['blocked']:
                    step_to_delete = min(50, len(self._data_list))
                    del self._data_list[-step_to_delete:]
                    log.warning(f'blocked, len={len(self._data_list)}')

                if terminal_debug['route_deviation']:
                    step_to_delete = min(50, len(self._data_list))
                    del self._data_list[-step_to_delete:]
                    log.warning(f'route deviation, len={len(self._data_list)}')

            if len(self._data_list) < 100:
                valid = False
            else:
                valid = True
            log.warning(f'Critical sampling start, len={len(self._data_list)}')

            def is_critical(x):
                c_t = (dagger_thresholds.throttle is not None) and (x[0] > dagger_thresholds.throttle)
                c_s = (dagger_thresholds.steer is not None) and (x[1] > dagger_thresholds.steer)
                c_b = (dagger_thresholds.brake is not None) and (x[2] > dagger_thresholds.brake)
                c_a = (dagger_thresholds.acc is not None) and (x[0]+x[2] > dagger_thresholds.acc)
                return c_t or c_s or c_b or c_a

            n_critical_frames = len(self._data_list)
            for idx, data in enumerate(self._data_list):
                if not is_critical(data['control_diff']):
                    self._data_list[idx]['critical'] = False
                    n_critical_frames -= 1
            log.warning(f'Critical sampling finished, n_critical_frames={n_critical_frames}')
        else:
            # behavior cloning dataset
            valid = True
            if remove_final_steps:
                if terminal_debug['traffic_rule_violated']:
                    step_to_delete = min(300, len(self._data_list))
                    del self._data_list[-step_to_delete:]
                    if len(self._data_list) < 300:
                        valid = False
                    log.warning(f'traffic_rule_violated, valid={valid}, len={len(self._data_list)}')

                if terminal_debug['blocked']:
                    step_to_delete = min(600, len(self._data_list))
                    del self._data_list[-step_to_delete:]
                    if len(self._data_list) < 300:
                        valid = False
                    log.warning(f'blocked, valid={valid}, len={len(self._data_list)}')

            if terminal_debug['route_deviation']:
                valid = False
                log.warning(f'route deviation, valid={valid}')

        # write to h5
        if valid:
            do_save = np.zeros(len(self._data_list), dtype=np.bool)
            for idx, data in enumerate(self._data_list):
                if data['critical']:
                    do_save[np.maximum(0, idx+self._im_stack_idx+1)] = True

            log.info(f'Saving {self._file_path}, data_len={len(self._data_list)}, save_len={do_save.sum()}')
            hf = h5py.File(self._file_path, 'w')
            for idx, data in enumerate(self._data_list):
                if do_save[idx]:
                    group_step = hf.create_group(f"step_{data['step']}")

                    group_step.attrs['critical'] = data['critical']

                    if last_value is not None:
                        if idx == len(self._data_list)-1:
                            next_value = last_value
                        else:
                            next_value = self._data_list[idx+1]['supervision']['value']
                        data['supervision']['advantage'] = data['reward'] + next_value - data['supervision']['value']

                    group_obs = group_step.create_group('obs')

                    for k, v in data['obs'].items():
                        if v is not None:
                            self._write_dict_to_group(group_obs, k, v)

                    # supervision
                    self._write_dict_to_group(group_step, 'supervision', data['supervision'])

                    # control_diff
                    if data['control_diff'] is not None:
                        group_step.create_dataset('control_diff', data=data['control_diff'])

            hf.close()
            hf = None
        self._data_list.clear()
        return valid
