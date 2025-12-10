from gmres import gmres
from bcg  import bcg
from bicgstab import bicgstab
from visualize import plot_residuals
from network import Network
import numpy as np
from scipy import linalg
import time

A, b = Network.initialize_network(5.0)

# quick direct solve to sanity-check
start = time.perf_counter()
x_direct = np.linalg.solve(A.toarray(), b)
direct_time = time.perf_counter() - start
direct_norm = np.linalg.norm(A @ x_direct - b)

# for name in network.nodes:
#     ind = network.node_ind[name]
#     print(f"{name:6s}: {x_direct[ind]: .6f}")

tol = 1e-15
max_iters = 100

print("residual norm direct:", direct_norm)
print("residual norm direct, inf:", np.linalg.norm(A @ x_direct - b,np.inf))
print("direct solve time:", direct_time)
print("----------------------------------------------------------")

# GMRES
start = time.perf_counter()
x_gmres, i, residuals_gmres = gmres(A,b, tol, max_iters)
gmres_time = time.perf_counter() - start
print("residual norm GMRES:", linalg.norm(A @ x_gmres - b))
print("residual norm GMRES, inf:", linalg.norm(A @ x_gmres - b,np.inf))
print("GMRES convergence iteration:", i)
print("GMRES solve time:", gmres_time)
print("----------------------------------------------------------")

# BCG
start = time.perf_counter()
x_bcg, i, residuals_bcg = bcg(A,b, tol, max_iters)
bcg_time = time.perf_counter() - start
print("residual norm BCG:", linalg.norm(A @ x_bcg - b))
print("residual norm BCG, inf:", linalg.norm(A @ x_bcg - b,np.inf))
print("BCG convergence iteration:", i)
print("BCG solve time:", bcg_time)
print("----------------------------------------------------------")

# BiCGStab
start = time.perf_counter()
x_bicgstab, i, residuals_bicgstab = bicgstab(A,b, tol, max_iters)
bicgstab_time = time.perf_counter() - start
print("residual norm BiCGStab:", linalg.norm(A @ x_bicgstab - b))
print("residual norm BiCGStab, inf:", linalg.norm(A @ x_bicgstab - b,np.inf))
print("BiCGStab convergence iteration:", i)
print("BiCGStab solve time:", bicgstab_time)
print("----------------------------------------------------------")

plot_residuals(residuals_gmres, residuals_bcg, residuals_bicgstab, direct_norm)