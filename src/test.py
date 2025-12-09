from network import Network
from vessel import Vessel
from gmres import gmres
import numpy as np
from scipy import linalg

# Define vessels

AA_in = Vessel("AA_in", "Heart", "AA", 1)

AA_CA = Vessel("AA_CA", "AA", "CA", 1)
CA_BR = Vessel("CA_BR", "CA", "BR", 1)
BR_SVC = Vessel("BR_SVC", "BR", "SVC", 1)
SVC_drain = Vessel("SVC_drain", "SVC", "drain", 1)

AA_DTAo = Vessel("AA_DTAo", "AA", "DTAo", 1)
DTAo_ADAo = Vessel("DTAo_ADAo", "DTAo", "ADAo", 1)
ADAo_LB = Vessel("ADAo_LB", "ADAo", "LB", 1)
LB_IVC = Vessel("LB_IVC", "LB", "IVC", 1)
IVC_drain = Vessel("IVC_drain", "IVC", "drain", 1)

vessels = [
    AA_in, 
    AA_CA, 
    CA_BR, 
    BR_SVC, 
    SVC_drain,
    AA_DTAo, 
    DTAo_ADAo, 
    ADAo_LB,
    LB_IVC, 
    IVC_drain
]

# Create network
network = Network(vessels, inlet="Heart", outlet="drain")
# Build Ax = b 
A, b = network.build_problem(Qin=10)

print("A shape: ", A.shape)
print("b: ", b)

# quick direct solve to sanity-check
x = np.linalg.solve(A.toarray(), b)

for name in network.nodes:
    ind = network.node_ind[name]
    print(f"{name:6s}: {x[ind]: .6f}")

print("residual norm direct:", np.linalg.norm(A @ x - b))

# GMRES
x_gmres, i = gmres(A,b, 1e-15, 50)
print("residual norm GMRES:", linalg.norm(A @ x_gmres - b))
print("residual norm GMRES, inf:", linalg.norm(A @ x_gmres - b,np.inf))
print("GMRES convergence iteration:", i)
