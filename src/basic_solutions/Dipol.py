
from .Solution import Solution
from typing import Tuple
import numpy as np

class Dipol(Solution):

    def __init__(self, mu: float, x_pos: float = 0, y_pos: float = 0) -> None:
        super().__init__(x_pos, y_pos)

        self.mu = mu

    def complex_potential(self, X: np.array, Y: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        z = (X - self.x_pos) + 1j*(Y - self.y_pos)

        F = self.mu / (2*np.pi) * 1 / z

        return F.real, F.imag


