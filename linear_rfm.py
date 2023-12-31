import numpy as np
import time
from numpy.linalg import solve, svd
np.set_printoptions(suppress = True)
from tqdm import tqdm

# Default implementation with matrix power alpha = 1
def linear_rfm(Y, unmasked, num_iters, reg=1e-1,
               return_out=False, early_stop=True,
               power=1, replace=True,
               MAX_PATIENCE=20):
    d, _ = Y.shape
    M = np.eye(d)

    sol = np.zeros((d, d))
    y = Y[unmasked].reshape(-1, 1)

    best_error = np.float('inf')
    prev_error = np.float('inf')
    patience = 0
    flag = False
    best_out = None

    for t in range(num_iters):
        n_obs = 0
        P = M.T @ M

        for i in range(d):
            idxs = np.nonzero(unmasked[i])[0]
            A = M[:, idxs]
            K = P[np.ix_(idxs, idxs)]
            a = solve(K + np.eye(len(K))*reg,
                      y[n_obs:n_obs+len(idxs)])

            sol[i, :] = (A @ a).T
            n_obs += len(idxs)

        out = sol @ M

        if replace:
            out[unmasked] = Y[unmasked]

        M = out.T @  out * 1/len(out)
        if power != 1:
            U, S, Vt = svd(M)
            S = np.where(S < 0., 0., S)
            S = np.power(S, power)
            M = U @ np.diag(S) @ Vt

        error = np.mean(np.square(Y[~unmasked] - out[~unmasked]))

        if abs(error - prev_error) < 1e-10 or error > prev_error:
            patience += 1
        if patience > MAX_PATIENCE:
            break
        if error < best_error:
            best_out = out
            best_error = error
        prev_error = error
        if (t+1) % 10 == 0 or t == 0:
            print("Round: ", t+1, "Error: ", error)
        if best_error < 1e-3 and early_stop:
            break
        if error < 1e-1:
            reg = min(reg, 1e-3)

    if return_out:
        return out, best_error
    else:
        return best_error
