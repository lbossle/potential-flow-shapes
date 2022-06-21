import numpy as np

class Element:

    def __init__(self, x: np.array, y: np.array, num_nodes: int, node_ids: np.array) -> None:

        self.x1 = x[0]
        self.x2 = x[1]
        self.x3 = x[2]
        self.x4 = x[3]

        self.y1 = y[0]
        self.y2 = y[1]
        self.y3 = y[2]
        self.y4 = y[3]

        self.a1 = 1/4 * (self.x1 - self.x2 - self.x3 + self.x4)
        self.a2 = 1/4 * (self.x1 - self.x2 + self.x3 - self.x4)
        self.a3 = 1/4 * (self.x1 + self.x2 - self.x3 - self.x4)

        
        self.b1 = 1/4 * (self.y1 - self.y2 - self.y3 + self.y4)
        self.b2 = 1/4 * (self.y1 - self.y2 + self.y3 - self.y4)
        self.b3 = 1/4 * (self.y1 + self.y2 - self.y3 - self.y4)

        self.stiffness_matrix = -1

        self.num_nodes = num_nodes
        self.node_ids = np.copy(node_ids)

    def shape_function_derivative(self, xi: float, eta: float) -> np.array:
        D = (self.a1+self.a2*xi)*(self.b3+self.b2*eta) - (self.a3+self.a2*eta)*(self.b1+self.b2*xi);
        B11 = 1/(4*D) * ((self.b3-self.b1)+(self.b2-self.b1)*xi+(self.b3-self.b2)*eta);
        B12 = 1/(4*D) * (-(self.b3+self.b1)-(self.b2-self.b1)*xi-(self.b3+self.b2)*eta);
        B13 = 1/(4*D) * (-(self.b3-self.b1)-(self.b2+self.b1)*xi+(self.b3+self.b2)*eta);
        B14 = 1/(4*D) * ((self.b3+self.b1)+(self.b2+self.b1)*xi-(self.b3-self.b2)*eta);

        B21 = 1/(4*D) * ((self.a1-self.a3)+(self.a1-self.a2)*xi+(self.a2-self.a3)*eta);
        B22 = 1/(4*D) * ((self.a1+self.a3)-(self.a1-self.a2)*xi+(self.a2+self.a3)*eta);
        B23 = 1/(4*D) * (-(self.a1-self.a3)+(self.a1+self.a2)*xi-(self.a2+self.a3)*eta);
        B24 = 1/(4*D) * (-(self.a1+self.a3)-(self.a1+self.a2)*xi-(self.a2-self.a3)*eta);

        B = np.array([[B11, B12, B13, B14], [B21, B22, B23, B24]])
        return B

    def calc_stiffness_matrix(self) -> np.array:
        def integrand(xi: float, eta: float) -> float:
            B = self.shape_function_derivative(xi, eta)
            result = np.array([B[0,:]]).T * B[0,:] + np.array([B[1,:]]).T * B[1,:]
            return result

        m = 2
        w = np.array([[1, 1], [1, 1]])
        x = np.array([0.5773502692, -0.5773502692])
        f = np.zeros([4, 4])
        for i in range(m):
            for j in range(m):
                f += w[i, j] * integrand(x[i], x[j])
        self.stiffness_matrix = f
        return f

    def get_assembly_matrix(self) -> np.array:
        L = np.zeros([4, self.num_nodes])
        L[0, self.node_ids[0]] = 1
        L[1, self.node_ids[1]] = 1
        L[2, self.node_ids[2]] = 1
        L[3, self.node_ids[3]] = 1
        return L.T @ self.stiffness_matrix @ L


if __name__ == "__main__":


    x = np.array([-1, 1, 1, -1])
    y = np.array([-1, -1, 1, 1])
    e = Element(x, y)
    print(e.calc_stiffness_matrix())
        
