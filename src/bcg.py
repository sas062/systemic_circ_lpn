from scipy import linalg
import numpy as np

def bcg(A, b, tol = 1e-10, max_iters = 50, return_residuals=True):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    residuals = [linalg.norm(r)]
    iters = 0

    s = r.copy()
    p = r.copy()
    q = s.copy()
    sr_old = s @ r

    for k in range(max_iters):
        if q @ A @ p == 0:
            break
        
        alpha = sr_old / (q @ A @ p)

        x += alpha * p
        r -= alpha * A @ p
        s -= alpha.conjugate() * A.T @ q

        residuals.append(linalg.norm(r))

        if residuals[-1] < tol:
            iters = k + 1
            break

        sr_new = s @ r
        beta = sr_new / sr_old

        p = r + beta * p
        q = s + beta.conjugate() * q

        sr_old = sr_new
        
    if return_residuals:
        return x, iters, residuals
    else:
        return x, max_iters