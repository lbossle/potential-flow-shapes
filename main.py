from src.basic_solutions.Dipol import Dipol
from src.basic_solutions.ParallelFlow import ParallelFlow
from src.fem.Domain import Domain
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # Analytical solution for cylinder
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


    ### Numerical solution for cylinder

    def condition1(x: float, y:float) -> bool:
        return x**2 + y**2 <= (11)**2

    def condition2(x: float, y: float) -> bool:
        return np.abs(x+y) + np.abs(x-y) >= 98
        
    def value1(x: float, y: float):
        return 0

    def value2(x: float, y: float):
        pFlow = ParallelFlow(1, 0)
        return pFlow.complex_potential(x, y)[1]

    domain = Domain()
    domain.read_mesh("mesh/cylinder.med")
    domain.create_elements()
    domain.plot_mesh()
    domain.assemble_K()
    domain.add_bcs(condition1, value1)
    domain.add_bcs(condition2, value2)
    domain.apply_bcs()
    domain.solve()
    domain.plot_solution()