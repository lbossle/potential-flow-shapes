
from .Solution import Solution
from typing import Tuple
import numpy as np

class Source(Solution):

    def __init__(self, M: float, x_pos: float = 0, y_pos: float = 0) -> None:
        super().__init__(x_pos, y_pos)

        self.M = M

    def complex_potential(self, X: np.array, Y: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        z = (X - self.x_pos) + 1j*(Y - self.y_pos)

        F = self.M / (2*np.pi) * np.log(z)

        return F.real, F.imag


