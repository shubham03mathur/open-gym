import numpy as np
from typing import Self, List, Deque
from collections import deque
class EMA:
    """ Class to calculate and return EMA of angle(s) """
    def __init__(self, angle: int | float, N = 3) -> None:
        if not isinstance(angle, (int, float)):
            raise Exception('Cannot process the request. Please use int or float only')
        self.alpha = 2 / (N + 1)
        radian = np.deg2rad(angle)
        self.ema_x = np.cos(radian)
        self.ema_y = np.sin(radian)
        self.angles = [np.rad2deg(radian).item()]
        self.timeseries = deque(maxlen=2)
        self.timeseries.append(np.rad2deg(radian).item())
    
    def update(self, angle: int | float) -> Self:
        """ Calculate EMA for given angle """
        rad = np.deg2rad(angle)
        x = np.cos(rad)
        y = np.sin(rad)
        self.ema_x = (x * self.alpha) + (self.ema_x * (1 - self.alpha))
        self.ema_y = (y * self.alpha) + (self.ema_y * (1 - self.alpha))

        angle = np.atan2(self.ema_y, self.ema_x)
        self.angles.append(np.degrees(angle).item())
        self.timeseries.append(np.degrees(angle).item())

        return self

    def get_ema(self) -> List[float]:
       """ Get EMA values """
       return self.angles

    def get_timeseries(self) -> Deque[float]:
        return self.timeseries

    def get_recent_ema(self) -> float:
        return self.angles.pop()
