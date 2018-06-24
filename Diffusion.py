import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

# data is an n x p matrix with rows as observations
# k is the number of neighbours (e.g. 20)
# d is the dimension of the embedding (e.g. 3)
# this implementation uses the zero-one kernel and diffusion time 0

def Diffusion( data, k, d ):

    # number of points
    n = data.shape[0];

    # neighbour search (replace with your favourite algorithm)
    knn = NearestNeighbors(n_neighbors=k).fit(data)
    _, jj = knn.kneighbors(data)
    ii = np.repeat(range(0, n), k)

    # affinity matrix
    W = sparse.coo_matrix((np.ones(n * k), (ii, jj.ravel())))
    W = W * W.transpose()

    # alpha normalization
    D = sparse.diags(1 / W.sum(axis=1).A.ravel())
    W = D * W * D

    # normalized graph laplacian
    D = sparse.diags(1 / np.sqrt(W.sum(axis=1)).A.ravel())
    P = D * W * D

    # eigendecomposition
    _, E = sparse.linalg.eigsh((P + P.transpose()) / 2, k=d+1, which='LM')

    # shift back and normalize
    E = D * E[:, :-1]
    # E = np.divide(E, np.linalg.norm(E, axis=0))

    return E

# E is an n x d matrix with rows as observations
# plot the last three columns of E for visualization purposes

# Written by John Malik on 2018.6.23, john.malik@duke.edu.
