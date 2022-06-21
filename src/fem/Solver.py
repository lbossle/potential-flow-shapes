from Element import Element
import numpy as np


if __name__ == "__main__":

    nx = 3
    ny = 3
    x = np.array([0., 1., 1., 0.])
    y= np.array([0., 0., 1., 1.])

    elements = []
    num_nodes = (nx+1) * (ny+1)
    node_ids = np.array([0, 1, nx+1, nx+2])

    for i in range(ny):
        for j in range(nx):
            element = Element(x, y, num_nodes, node_ids)
            element.calc_stiffness_matrix()
            elements.append(element)
            x += np.ones(4)
            node_ids += 1
        y += np.ones(4)

    K = np.zeros([num_nodes, num_nodes])

    for element in elements:
        K +=element.get_assembly_matrix()
    print(K)

    essential_nodes = [0, 1, 2, 3, 4, 8, 12, 15, 11, 7, 13, 14]
    d_E = np.array([0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 1, 2])
    L_E = np.zeros([num_nodes, len(essential_nodes)]) 
    for i in range(len(essential_nodes)):
        L_E[essential_nodes[i], i] = 1

    free_nodes = []
    for i in range(num_nodes):
        if i not in essential_nodes:
             free_nodes.append(i)

    L_F = np.zeros([num_nodes, len(free_nodes)])
    for i in range(len(free_nodes)):
        L_F[free_nodes[i], i] = 1

    K_FE = L_F.T @ K @ L_E
    K_FF = L_F.T @ K @ L_F

    # 0 = K_FE @ d_E + K_FF @ d_F 
    # 

    sol = np.linalg.solve(K_FF, -K_FE @ d_E)

    print(sol)