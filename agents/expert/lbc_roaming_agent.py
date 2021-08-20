# Modified from lbc/roaming_agent and carla/basic_agent
import numpy as np
import logging
import cv2
import carla
from omegaconf import OmegaConf
import copy

from .utils.local_planner import LocalPlanner
from carla_gym.utils.hazard_actor import lbc_hazard_vehicle, lbc_hazard_walker
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE_0 = (252, 175, 62)


class LbcRoamingAgent(object):
    def __init__(self, path_to_conf_file):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)

        self._obs_configs = OmegaConf.to_container(cfg.obs_configs)

        self._hack_throttle = cfg.hack_throttle

        resolution = cfg.resolution
        target_speed = cfg.target_speed
        longitudinal_pid_params = cfg.longitudinal_pid_params
        lateral_pid_params = cfg.lateral_pid_params
        threshold_before = cfg.threshold_before
        threshold_after = cfg.threshold_after
        self._local_planner = LocalPlanner(resolution, target_speed, longitudinal_pid_params,
                                           lateral_pid_params, threshold_before, threshold_after)

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

        # reset local planner
        self._local_planner.reset()

    def run_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)

        hazard_vehicle_loc = lbc_hazard_vehicle(input_data['surrounding_vehicles'], input_data['speed']['speed_xy'][0])
        vehicle_hazard = hazard_vehicle_loc is not None
        hazard_ped_loc = lbc_hazard_walker(input_data['surrounding_pedestrians'])
        pedestrian_ahead = hazard_ped_loc is not None
        redlight_ahead = input_data['traffic_light']['at_red_light'] == 1
        stop_sign_ahead = input_data['stop_sign']['at_stop_sign'] == 1
        negative_speed = input_data['speed']['forward_speed'] < -1.0

        # if vehicle_hazard or pedestrian_ahead:
        if vehicle_hazard or pedestrian_ahead or redlight_ahead or stop_sign_ahead or negative_speed:
            # emergency stop
            throttle, steer, brake = 0.0, 0.0, 1.0
        else:
            throttle, steer, brake = self._local_planner.run_step(
                input_data['route_plan'], input_data['speed']['speed_xy'][0])
            if self._hack_throttle:
                throttle *= max((1.0 - abs(steer)), 0.25)

        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

        # supervision_dict for IL
        self.supervision_dict = {
            'action': np.array([throttle, steer, brake], dtype=np.float32),
            'speed': input_data['speed']['forward_speed']
        }
        self.supervision_dict = copy.deepcopy(self.supervision_dict)

        self._render_dict = {
            'step': timestamp['step'],
            'birdview': input_data['birdview']['rendered'],
            'route_plan': input_data['route_plan']['location'],
            'command': input_data['route_plan']['command'],
            'speed': input_data['speed']['speed_xy'][0],
            'control': np.array([throttle, steer, brake], dtype=np.float64),
            'vehicle_hazard': vehicle_hazard,
            'pedestrian_ahead': pedestrian_ahead,
            'redlight_ahead': redlight_ahead,
            'stop_sign_ahead': stop_sign_ahead,
            'gnss': input_data['gnss']
        }
        self._render_dict = copy.deepcopy(self._render_dict)

        return control

    def render(self, reward_debug, terminal_debug):
        im_birdview = self.draw_route(self._render_dict, self._obs_configs)
        im_birdview = self.draw_gnss(im_birdview, self._render_dict, self._obs_configs)

        h, w, c = im_birdview.shape
        im = np.zeros([h, w*2, c], dtype=np.uint8)
        im[:h, :w] = im_birdview

        speed = self._render_dict['speed']
        step = self._render_dict['step']

        txt_1 = f'n:{step:5} spd:{speed:5.2f} comp:{self._render_dict["gnss"]["imu"][-1]:.2f}'
        im = cv2.putText(im, txt_1, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        throttle, steer, brake = self._render_dict['control']
        txt_2 = f's:{steer:5.2f}, t:{throttle:5.2f}, b:{brake:5.2f}'
        im = cv2.putText(im, txt_2, (2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        vehicle_hazard = int(self._render_dict['vehicle_hazard'])
        pedestrian_ahead = int(self._render_dict['pedestrian_ahead'])
        redlight_ahead = int(self._render_dict['redlight_ahead'])
        stop_sign_ahead = int(self._render_dict['stop_sign_ahead'])
        txt_3 = f'vh:{vehicle_hazard}, pd:{pedestrian_ahead}, tl:{redlight_ahead}, ss:{stop_sign_ahead}'
        im = cv2.putText(im, txt_3, (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)

        imu_str = np.array2string(self._render_dict['gnss']['imu'][:-1],
                                  precision=1, separator=',', suppress_small=True)
        im = cv2.putText(im, imu_str, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        for i, txt in enumerate(reward_debug['debug_texts'] + terminal_debug['debug_texts']):
            im = cv2.putText(im, txt, (w, 12+(i+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return im

    @staticmethod
    def draw_gnss(rendered_birdview, render_dict, obs_cfg):
        birdview_cfg = obs_cfg['birdview']

        ev_gps = render_dict['gnss']['gnss']
        compass = 0.0 if np.isnan(render_dict['gnss']['imu'][-1]) else render_dict['gnss']['imu'][-1]

        gps_point = render_dict['gnss']['target_gps']
        target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)

        loc = [loc_in_ev.x, loc_in_ev.y]
        ev_xy = (int(birdview_cfg['width_in_pixels']/2),
                 int(birdview_cfg['width_in_pixels']-birdview_cfg['pixels_ev_to_bottom']))

        x = int(np.round(ev_xy[0] + loc[1]*birdview_cfg['pixels_per_meter']))
        y = int(np.round(ev_xy[1] - loc[0]*birdview_cfg['pixels_per_meter']))

        cmd = render_dict['gnss']['command'][0]

        if cmd == 1:
            # LEFT = 1
            color = COLOR_RED
        elif cmd == 2:
            # RIGHT = 2
            color = COLOR_GREEN
        elif cmd == 3:
            # STRAIGHT = 3
            color = COLOR_ORANGE_0
        elif cmd == 4:
            # LANEFOLLOW = 4
            color = COLOR_WHITE
        elif cmd == 5:
            # CHANGELANELEFT = 5
            color = COLOR_YELLOW
        elif cmd == 6:
            # CHANGELANERIGHT = 6
            color = COLOR_BLUE
        elif cmd == -1:
            # VOID = -1
            color = COLOR_BLACK
        else:
            assert f'error {cmd}'

        cv2.line(rendered_birdview, ev_xy, (x, y), color=color, thickness=2)

        return rendered_birdview

    @staticmethod
    def draw_route(render_dict, obs_cfg):
        birdview_cfg = obs_cfg['birdview']
        rendered_birdview = render_dict['birdview'].copy()

        for i, loc in enumerate(render_dict['route_plan']):
            x = int(np.round(birdview_cfg['width_in_pixels']/2 + loc[1]*birdview_cfg['pixels_per_meter']))
            y = int(np.round(birdview_cfg['width_in_pixels'] - birdview_cfg['pixels_ev_to_bottom']
                             - loc[0] * birdview_cfg['pixels_per_meter']))

            # VOID = 0
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 3
            # LANEFOLLOW = 4
            # CHANGELANELEFT = 5
            # CHANGELANERIGHT = 6
            cmd = render_dict['command'][i]

            if cmd == 1:
                # LEFT = 1
                color = COLOR_RED
            elif cmd == 2:
                # RIGHT = 2
                color = COLOR_GREEN
            elif cmd == 3:
                # STRAIGHT = 3
                color = COLOR_ORANGE_0
            elif cmd == 4:
                # LANEFOLLOW = 4
                color = COLOR_WHITE
            elif cmd == 5:
                # CHANGELANELEFT = 5
                color = COLOR_YELLOW
            elif cmd == 6:
                # CHANGELANERIGHT = 6
                color = COLOR_BLUE
            elif cmd == -1:
                # VOID = -1
                color = COLOR_BLACK
            else:
                print('error!!!!', cmd)
            cv2.circle(rendered_birdview, (x, y), 3, color, -1)

        return rendered_birdview

    @property
    def obs_configs(self):
        return self._obs_configs
