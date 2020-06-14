import enum

import numpy as np
from scipy.stats import norm


class Detector:

    def __init__(self, buffer_size=100):
        self.buffer_real_values = []
        self.buffer_prediction_values = []
        self.buffer_size = buffer_size

    def add_values(self, y, y_hat):
        self.buffer_real_values.append(y)
        self.buffer_prediction_values.append(y_hat)
        self.update_buffer()

    def update_buffer(self):
        if len(self.buffer_real_values) > self.buffer_size:
            self.buffer_real_values = self.buffer_real_values[-self.buffer_size:]
            self.buffer_prediction_values = self.buffer_prediction_values[-self.buffer_size:]

    def reset(self):
        self.buffer_real_values = []
        self.buffer_prediction_values = []


class GaussianTailDetector(Detector):

    def __init__(self, window_size, small_window_size, e=0.3, buffer_size=100):
        Detector.__init__(self, buffer_size)

        self.window_size = window_size
        self.small_window_size = small_window_size
        self.anomaly_scores = []
        self.e = e

    def is_anomaly(self):
        self.anomaly_scores.append(max(abs(self.buffer_prediction_values[-1] - self.buffer_real_values[-1]), 0))
        if len(self.anomaly_scores) < self.window_size:
            return False
        window = self.anomaly_scores[-self.window_size:]
        small_window = self.anomaly_scores[-self.small_window_size:]
        std = np.std(window)
        mean = np.mean(window)
        distribution = norm(mean, std)
        q_value = np.mean(distribution.sf(np.array(small_window)))
        L = 1 - q_value

        return L >= (1 - self.e)

    def reset(self):
        super().reset()
        self.anomaly_scores = []


class DiffType(enum.Enum):
    abs = 1
    up = 2
    down = 3


class AccumulativeDetector(Detector):

    def __init__(self, t=2.5, e=-0.1, var_window_size=50, with_var=False, buffer_size=100,
                 diff_type=DiffType.abs):
        Detector.__init__(self, buffer_size)

        self.a = 0.
        self.t = t
        self.e = e
        self.var_window_size = var_window_size
        self.with_var = with_var
        self.diff_type = diff_type
        self.peak_reached = False

    def is_anomaly(self) -> bool:
        if self.peak_reached:
            self.a = 0.
            self.peak_reached = False

        t = self.t
        if self.with_var:
            if len(self.buffer_real_values) < self.var_window_size:
                return False
            t = np.std(self.buffer_real_values[-self.var_window_size]) * 20

        y_val = self.buffer_real_values[-1]
        y_hat_val = self.buffer_prediction_values[-1]

        diff = self.get_diff(y_val, y_hat_val)

        if diff + self.e > 0:
            if self.a < 1.5 * t:
                self.a += diff
            else:
                self.peak_reached = True
        else:
            if self.a > 0:
                self.a -= 0.30
        return self.a > t

    def get_diff(self, y_val, y_hat_val):
        if self.diff_type == DiffType.abs:
            return abs(y_val - y_hat_val)
        if self.diff_type == DiffType.up:
            return y_val - y_hat_val
        if self.diff_type == DiffType.down:
            return y_hat_val - y_val

    def reset(self):
        super().reset()
        self.a = 0.
        self.peak_reached = False
