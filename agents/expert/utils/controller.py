from collections import deque
import numpy as np

class PIDController(object):
    def __init__(self, pid_list, n=30):
        self._K_P, self._K_I, self._K_D = pid_list

        self._dt = 1.0 / 10.0
        self._window = deque(maxlen=n)

    def reset(self):
        self._window.clear()

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = sum(self._window) * self._dt
            derivative = (self._window[-1] - self._window[-2]) / self._dt
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control

class PIDControllerAutopilot(object):
    def __init__(self, pid_list, n=30):
        self._K_P, self._K_I, self._K_D = pid_list

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def reset(self):
        self._window.clear()

    def step(self, error):
        error = 0.5 * error # leaderboard uses 20 FPS, we use 10 FPS
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        control = 0.0
        control += self._K_P * error
        control += self._K_I * integral
        control += self._K_D * derivative

        return control