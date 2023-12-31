import numpy as np
from numpy.linalg import solve
import time
np.set_printoptions(suppress = True)

# SVD free implementation of lin-RFM with power alpha = 1/2
def rfm(Y, unmasked, num_iters, reg=1e-1,
        return_out=False, early_stop=True,
        replace=True, MAX_PATIENCE=20):

    d, _ = Y.shape
    M = np.eye(d)
    y = Y[unmasked].reshape(-1, 1)
    sol = np.zeros((d,d))

    best_error = np.float('inf')
    prev_error = np.float('inf')
    patience = 0
    flag = False
    best_out = None

    for t in range(num_iters):
        n_obs = 0
        new_M = 0
        for i in range(d):
            idxs = np.nonzero(unmasked[i])[0]
            K = M[np.ix_(idxs, idxs)]
            R = np.eye(len(K)) * reg
            a = solve(K + R, y[n_obs:n_obs+len(idxs)]).T
            sol_piece = a @ M[idxs, :]
            if replace:
                sol_piece[:, idxs] = Y[i, idxs]
            sol[i, :] = sol_piece
            n_obs += len(idxs)

        out = sol
        M = out.T @  out * 1/len(out)

        error = np.mean(np.square(Y[~unmasked] - out[~unmasked]))

        if abs(error - prev_error) < 1e-10 or error > prev_error:
            patience += 1
        if patience > MAX_PATIENCE:
            break
        if error < best_error:
            best_out = out
            best_error = error
        prev_error = error
        if (t + 1) % 10 == 0:
            print("Round: ", t+1, "Error: ", error)
        if best_error < 1e-3 and early_stop:
            break
        if error < 1e-1:
            reg = min(reg, 1e-3)

    if return_out:
        return out, best_error
    else:
        return best_error
