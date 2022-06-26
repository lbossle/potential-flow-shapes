from typing import Callable
import meshio
import numpy as np
from .Element import Element
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class Domain:

    def __init__(self) -> None:
        self.num_nodes = 0
        self.elements = []
        self.essential_nodes = np.array([], dtype=int)
        self.d_E = np.array([], dtype=float)

    def read_mesh(self, filename: str) -> None:
        self.mesh = meshio.read(filename)
        self.cells = self.mesh.cells[0].data
        self.points = self.mesh.points
        self.num_nodes = len(self.points)
        self.K = np.zeros([self.num_nodes, self.num_nodes])

    def create_elements(self):
        for cell in self.cells:
            x = np.array([self.points[cell[0]][0], self.points[cell[1]][0], self.points[cell[2]][0], self.points[cell[3]][0]])
            y = np.array([self.points[cell[0]][1], self.points[cell[1]][1], self.points[cell[2]][1], self.points[cell[3]][1]])
            self.elements.append(Element(x, y, self.num_nodes, node_ids=cell))

    def assemble_K(self):
        self.K = np.zeros([self.num_nodes, self.num_nodes])
        for element in self.elements:
            element.calc_stiffness_matrix()
            self.K += element.get_assembly_matrix()

    def add_bcs(self, condition: Callable, value: float):
        for node_id, point in enumerate(self.points):
            if condition(point[0], point[1]):
                #plt.plot(point[0], point[1], "ko")
                self.essential_nodes = np.append(self.essential_nodes, node_id)
                self.d_E = np.append(self.d_E, value(point[0], point[1]))
        #plt.show()

    def apply_bcs(self):
        self.L_E = np.zeros([self.num_nodes, len(self.essential_nodes)], dtype=int) 
        for i in range(len(self.essential_nodes)):
            self.L_E[self.essential_nodes[i], i] = 1

        self.free_nodes = []
        for i in range(self.num_nodes):
            if i not in self.essential_nodes:
                self.free_nodes.append(i)
        
        self.L_F = np.zeros([self.num_nodes, len(self.free_nodes)])
        for i in range(len(self.free_nodes)):
            self.L_F[self.free_nodes[i], i] = 1

    def solve(self):
        K_FE = self.L_F.T @ self.K @ self.L_E
        K_FF = self.L_F.T @ self.K @ self.L_F
        self.solution = np.linalg.solve(K_FF, -K_FE @ self.d_E)
        return self.solution

    def plot_solution(self):
        x = []
        y = []
        val = []

        for id, fn_id in enumerate(self.free_nodes):
            x.append(self.points[fn_id][0])
            y.append(self.points[fn_id][1])
            val.append(self.solution[id])

        for id, en_id in enumerate(self.essential_nodes):
            x.append(self.points[en_id][0])
            y.append(self.points[en_id][1])
            val.append(self.d_E[id])

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,val)
        plt.show()

        xi = np.linspace(-50, 50, 500)
        yi = np.linspace(-50, 50, 500)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(x, y)
        interpolator = tri.LinearTriInterpolator(triang, val)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        ax.contour(xi, yi, zi, levels=50)
        ax.contour(xi, yi, zi, levels=[-1e-6, 1e-6], colors="red", linestyles="solid")
        
        plt.show()

            
    def plot_mesh(self):
        fig, ax = plt.subplots()
        ax.set_box_aspect(1)
        for node in self.points:
            ax.plot(node[0], node[1], "ko")
        ax.grid()
        plt.show()