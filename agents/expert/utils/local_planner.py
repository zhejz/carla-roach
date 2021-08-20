from enum import Enum
import numpy as np

from .controller import PIDController


class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations
    when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class LocalPlanner(object):

    def __init__(
            self, resolution, target_speed, longitudinal_pid_params, lateral_pid_params, threshold_before=7.5,
            threshold_after=5.0):
        self._target_speed = target_speed

        self._speed_pid = PIDController(longitudinal_pid_params)
        self._turn_pid = PIDController(lateral_pid_params)

        self._resolution = resolution
        self._threshold_before = threshold_before
        self._threshold_after = threshold_after
        self._max_skip = 20

        self._last_command = 4

    def reset(self):
        self._last_command = 4
        self._speed_pid.reset()
        self._turn_pid.reset()

    def run_step(self, current_plan, ego_vehicle_speed):
        target_index = -1
        for i, (location, command) in enumerate(zip(current_plan['location'], current_plan['command'])):
            if i > self._max_skip:
                break

            if self._last_command == 4 and command != 4:
                threshold = self._threshold_before
            else:
                threshold = self._threshold_after

            distance = np.linalg.norm(location[0:2])
            if distance < threshold:
                self._last_command = command
                target_index = i

        target_index += 1
        if target_index == len(current_plan['location']):
            target_index = len(current_plan['location']) - 1
        target_location = current_plan['location'][target_index]
        target_command = current_plan['command'][target_index]

        # steer
        x = target_location[0]
        y = target_location[1]
        theta = np.arctan2(y, x)
        steer = self._turn_pid.step(theta)

        # throttle
        target_speed = self._target_speed
        if target_command not in [3, 4]:
            target_speed *= 0.75
        delta = target_speed - ego_vehicle_speed
        throttle = self._speed_pid.step(delta)

        # brake
        brake = 0.0

        # clip
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)

        return throttle, steer, brake
