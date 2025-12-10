import numpy as np
from scipy.sparse import csr_matrix, diags
from vessel import Vessel

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
    
    def initialize_network(Qin):
    
        # Define vessels
        AA_in = Vessel("AA_in", "Heart", "AA", 0.6)

        AA_CA = Vessel("AA_CA", "AA", "CA", 0.6)
        Ca_cerebral = Vessel("Ca_cerebral", "CA", "cerebral", 2.5)
        cerebral_SVC = Vessel("cerebral_SVC", "cerebral", "SVC", 0.2)
        # CA_BR = Vessel("CA_BR", "CA", "BR", 1.5)
        AA_BR = Vessel("AA_BR", "AA", "BR", 0.6)
        BR_arms = Vessel("BR_arms", "BR", "arms", 2.0)
        arms_SVC = Vessel("arms_SVC", "arms", "SVC", 0.2)
        # BR_SVC = Vessel("BR_SVC", "BR", "SVC", 0.2)
        SVC_drain = Vessel("SVC_drain", "SVC", "drain", 0.05)

        AA_cor = Vessel("AA_cor", "AA", "coronary", 0.4)
        cor_hrt = Vessel("cor_hrt", "coronary", "heart", 3.0)
        hrt_drain = Vessel("hrt_drain", "heart", "drain", 0.05)

        AA_DTAo = Vessel("AA_DTAo", "AA", "DTAo", 0.6)
        DTAo_ADAo = Vessel("DTAo_ADAo", "DTAo", "ADAo", 1.2)
        ADAo_LB = Vessel("ADAo_LB", "ADAo", "LB", 1.5)
        LB_muscles = Vessel("LB_muscles", "LB", "muscles", 2.0)
        muscles_IVC = Vessel("muscles_IVC", "muscles", "IVC", 0.2)
        # LB_IVC = Vessel("LB_IVC", "LB", "IVC", 0.2)
        ADAo_Renal = Vessel("ADAo_Renal", "ADAo", "Renal", 0.8)
        Renal_KID = Vessel("Renal_KID", "Renal", "KID", 3.0)
        KID_IVC = Vessel("KID_IVC", "KID", "IVC", 0.2)
        DTAo_Hepatic = Vessel("DTAo_Hepatic", "DTAo", "Hepatic", 0.8)
        Hepatic_LIV = Vessel("Hepatic_LIV", "Hepatic", "LIV", 3.5)
        LIV_IVC = Vessel("LIV_IVC", "LIV", "IVC", 0.2)
        IVC_drain = Vessel("IVC_drain", "IVC", "drain", 0.05)

        vessels = [
            AA_in, 
            AA_CA, 
            Ca_cerebral, 
            cerebral_SVC,
            # CA_BR, 
            AA_BR,
            BR_arms,
            arms_SVC,
            # BR_SVC,
            SVC_drain,
            AA_cor, 
            cor_hrt,
            hrt_drain,
            AA_DTAo, 
            DTAo_ADAo, 
            ADAo_LB,
            LB_muscles, 
            muscles_IVC,
            # LB_IVC,
            ADAo_Renal,
            Renal_KID,
            KID_IVC,
            DTAo_Hepatic,
            Hepatic_LIV,
            LIV_IVC,
            IVC_drain
        ]

        # Create network
        network = Network(vessels, inlet="Heart", outlet="drain")
        # Build Ax = b 
        A, b = network.build_problem(Qin=Qin)

        print("A shape: ", A.shape)
        print("b: ", b)
        print("----------------------------------------------------------")

        return A, b
