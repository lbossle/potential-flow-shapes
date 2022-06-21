
from .Solution import Solution
from typing import Tuple
import numpy as np


class ParallelFlow(Solution):

    def __init__(self, U_inf: float, alpha: float, x_pos: float = 0, y_pos: float = 0) -> None:
        super().__init__(x_pos, y_pos)

        self.U_inf = U_inf
        self.alpha = alpha

    def complex_potential(self, X: np.array, Y: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        z = (X - self.x_pos) + 1j*(Y - self.y_pos)

        F = self.U_inf * (np.cos(self.alpha) - 1j*np.sin(self.alpha)) * z

        return F.real, F.imag