
from .Solution import Solution
from typing import Tuple
import numpy as np

class Vortex(Solution):

    def __init__(self, gamma: float, x_pos: float = 0, y_pos: float = 0) -> None:
        super().__init__(x_pos, y_pos)

        self.gamma = gamma

    def complex_potential(self, X: np.array, Y: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        z = (X - self.x_pos) + 1j*(Y - self.y_pos)

        F = -1j * self.gamma / (2*np.pi) * np.log(z)

        return F.real, F.imag


