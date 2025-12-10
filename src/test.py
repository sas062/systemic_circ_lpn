from network import Network
from vessel import Vessel
from gmres import gmres
from bcg  import bcg
from bicgstab import bicgstab
# from visualize import plot_residuals
import numpy as np
from scipy import linalg

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
A, b = network.build_problem(Qin=5)

print("A shape: ", A.shape)
print("b: ", b)

# quick direct solve to sanity-check
x_direct = np.linalg.solve(A.toarray(), b)
direct_norm = np.linalg.norm(A @ x_direct - b)

for name in network.nodes:
    ind = network.node_ind[name]
    print(f"{name:6s}: {x_direct[ind]: .6f}")

print("residual norm direct:", direct_norm)
print("residual norm direct, inf:", np.linalg.norm(A @ x_direct - b,np.inf))
print("----------------------------------------------------------")

# GMRES
x_gmres, i, residuals_gmres = gmres(A,b, 1e-15, 50)
print("residual norm GMRES:", linalg.norm(A @ x_gmres - b))
print("residual norm GMRES, inf:", linalg.norm(A @ x_gmres - b,np.inf))
print("GMRES convergence iteration:", i)
print("----------------------------------------------------------")


# BCG
x_bcg, i, residuals_bcg = bcg(A,b, 1e-15, 50)
print("residual norm BCG:", linalg.norm(A @ x_bcg - b))
print("residual norm BCG, inf:", linalg.norm(A @ x_bcg - b,np.inf))
print("BCG convergence iteration:", i)
print("----------------------------------------------------------")


# BiCGStab
x_bicgstab, i, residuals_bicgstab = bicgstab(A,b, 1e-15, 50)
print("residual norm BiCGStab:", linalg.norm(A @ x_bicgstab - b))
print("residual norm BiCGStab, inf:", linalg.norm(A @ x_bicgstab - b,np.inf))
print("BiCGStab convergence iteration:", i)
print("----------------------------------------------------------")

# plot_residuals(residuals_gmres, residuals_bcg, residuals_bicgstab, direct_norm)