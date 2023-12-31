import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


# d is shape of matrix
# r is rank
def get_data(d, r, num_obs):

    U = np.random.normal(size=(d,r))
    V = np.random.normal(size=(r,d))
    Y = U @ V / np.sqrt(r)
    Y = Y / np.linalg.norm(Y, 'fro') * d

    mask = np.full(d*d, True)
    mask[:num_obs] = False

    np.random.shuffle(mask)
    mask = mask.reshape(d,d)
    unmasked = ~mask

    return Y, unmasked
