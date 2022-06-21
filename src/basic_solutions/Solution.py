
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class Solution(ABC):

    def __init__(self, x_pos: float = 0, y_pos: float = 0) -> None:
        super().__init__()

        self.x_pos = x_pos
        self.y_pos = y_pos

    @abstractmethod
    def complex_potential(self, X: np.array, Y: np.array, *args, **kwargs) -> Tuple[np.array, np.array]:
        pass