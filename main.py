from src.basic_solutions.Dipol import Dipol
from src.basic_solutions.ParallelFlow import ParallelFlow
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y, indexing='ij')


    phiP, psiP = ParallelFlow(1, 0).complex_potential(X, Y)
    phiD, psiD = Dipol(1,  y_pos = 0).complex_potential(X, Y)


    fig1, ax = plt.subplots()
    ax.set_box_aspect(1)

    ax.contour(X, Y, psiP + psiD, levels=n)
    plt.show()