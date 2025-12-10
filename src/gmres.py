from scipy import linalg
import numpy as np

def gmres(A, b, tol = 1e-10, max_iters = 50, return_residuals=True):
    n = len(b)
    x = np.zeros(n)

    Q = np.zeros((n, max_iters+1))
    H = np.zeros((max_iters+1,max_iters))
    
    residuals = []

    k = 0
    
    for i in range(max_iters):
        r = b - A @ x
        residuals.append(linalg.norm(r))

        if residuals[-1] < tol:
            k = i + 1
            break
        
        if i == 0:
            Q[:, i] = r / linalg.norm(r)

        v = A @ Q[:, i]

        for j in range (i+1):
            H[j,i] = Q[:, j] @ v
            v -= H[j,i] * Q[:, j]

        H[i+1,i] = linalg.norm(v)

        # Breakdown check
        if H[i+1,i] == 0:
            k = -1
            break

        Q[:, i+1] = v / H[i+1,i]
        
        k = i + 1

        e1 = np.zeros(k+1)
        e1[0] = 1.0

        y, *_ = linalg.lstsq(H[:k+1,:k], linalg.norm(b)*e1)

        x = Q[:,:k] @ y

    if return_residuals:
        return x, k, residuals
    else:
        return x, k