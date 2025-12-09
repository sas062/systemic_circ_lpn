from scipy import linalg
import numpy as np

def bicgstab(A, b, tol = 1e-10, max_iters = 50):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    r_hat = r.copy()
    rho_old = alpha = omega = 1.0
    v = p = np.zeros(n)

    for k in range(max_iters):
        rho_new = r_hat @ r
        if rho_new == 0:
            break

        if k == 0:
            p = r
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

        v = A @ p
        alpha = rho_new / (r_hat @ v)
        s = r - alpha * v

        if linalg.norm(s) < tol:
            x += alpha * p
            return x, k+1

        t = A @ s
        omega = (t @ s) / (t @ t)
        x += alpha * p + omega * s
        r = s - omega * t

        if linalg.norm(r) < tol:
            return x, k+1

        rho_old = rho_new

    return x, max_iters