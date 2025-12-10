import numpy as np
from scipy.sparse import csr_matrix, diags

class Network:
    def __init__(self, vessels, inlet, outlet):
        self.vessels = vessels
        self.inlet = inlet
        self.outlet = outlet

        nodes = set()
        for v in vessels:
            nodes.add(v.start_node)
            nodes.add(v.end_node)
        self.nodes = sorted(nodes)
        self.node_ind = {name: i for i, name in enumerate(self.nodes)}
    
    def build_problem(self, Qin):
        rows = []
        cols = []
        data = []
        m = len(self.vessels)
        n = len(self.nodes)

        for k, v in enumerate(self.vessels):
            i = self.node_ind[v.start_node]
            j = self.node_ind[v.end_node]

            rows.append(i); cols.append(k); data.append(-1.0)
            rows.append(j); cols.append(k); data.append(1.0)

        B = csr_matrix((data, (rows, cols)), shape=(n,m))
        
        G = diags(1.0 / np.array([v.R for v in self.vessels]))

        A = B @ G @ B.T

        b = np.zeros(n)
        b[self.node_ind[self.inlet]] = Qin
        b[self.node_ind[self.outlet]] = 0.0

        drain_ind = self.node_ind[self.outlet]
        A = A.tolil()
        A[drain_ind, :] = 0
        A[drain_ind, drain_ind] = 1.0
        A = A.tocsr()

        return A, b
