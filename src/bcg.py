from scipy import linalg
import numpy as np

def bcg(A, b, tol = 1e-10, max_iters = 50):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x

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

        if linalg.norm(r) < tol:
            return x, k+1

        sr_new = s @ r
        beta = sr_new / sr_old

        p = r + beta * p
        q = s + beta.conjugate() * q

        sr_old = sr_new
    
    return x, max_iters