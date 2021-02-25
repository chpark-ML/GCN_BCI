import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import numpy as np
import tensorflow as tf

def distance_scipy_spatial(z, k=4, metric='euclidean', thresh = 0):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)

    # Calculate the maximum distance and set the threshold
    # dist_max = np.amax(d) # float64
    # thresh = dist_max / 2 # float64

    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]

    return d, idx


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    """Calculate the minimum distance and make them zero"""
    dist_min = np.amin(dist) # float64
    row_dist, col_dist = dist.shape
    for i in range(row_dist):
        for j in range (col_dist):
            if dist[i][j] == dist_min :
                dist[i][j] = 0

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T >= W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def laplacian_tf(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""
    N = W.get_shape().as_list()
    # Degree matrix.
    # d = W.sum(axis=0)
    D = tf.reduce_sum(W, axis=0)

    # Laplacian matrix.
    if not normalized:
        D_ = tf.diag(D)
        # D = scipy.sparse.diags(d.A.squeeze(), 0)
        # L = D_ - W
        L = tf.subtract(D_, W)
    else:
        # d += np.spacing(np.array(0, W.dtype))
        # d = 1 / np.sqrt(d)
        # D = scipy.sparse.diags(d.A.squeeze(), 0)
        # I = scipy.sparse.identity(d.size, dtype=W.dtype)
        # L = I - D * W * D
        tf.add(D ,tf.constant(np.spacing(np.array(0, np.float32))))
        # D_ = 1.0 / tf.sqrt(D)
        D_ = tf.divide(1.0 , tf.sqrt(D))
        D_ = tf.diag(D_)
        I = tf.eye(num_rows=N[0], num_columns=N[1])
        L = tf.subtract(I ,tf.matmul(D_,tf.matmul(W,D_)))

    # assert np.abs(L - L.T).mean() < 1e-9
    # assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def plot_spectrum(L, algo='eig'):
    """Plot the spectrum of a list of multi-scale Laplacians L."""
    # Algo is eig to be sure to get all eigenvalues.
    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = fourier(lap, algo)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)


def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo is 'eigs':
        lamb, U = scipy.sparse.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = scipy.sparse.linalg.eigsh(L, k=k, which='SM')

    return lamb, U

def rescale_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    M, M = L.shape
    I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    L /= lmax / 2
    L -= I
    return L

def rescale_tf_L(L, lmax=2):
    """Rescale the Laplacian eigenvalues in [-1,1]."""
    # M, M = L.shape
    M = L.get_shape().as_list()
    # I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
    # I = tf.identity(L)
    I = tf.eye(num_rows=M[0])

    # L /= lmax / 2
    tf.divide(L, lmax / 2)
    tf.subtract(L, I)
    # L -= I
    return L


def AdjFromIllust():
    ill = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 2, 3, 4, 5, 6, 0],
        [7,8,9,10,11,12,13],
        [0,14,15,16,17,18,0],
        [0,0,19, 20, 21,0,0],
        [0,0,0,22,0,0,0]
        ])


    adj = np.zeros([22, 22], dtype=np.float32)

    for i in range(6):  # row
        for j in range(7):  # collum

            if ill[i, j] != 0 :
                ## Check the (i-1, j) has the connection
                if i - 1 >= 0 :
                    if ill[i - 1, j] != 0:
                        adj[ill[i, j] - 1, ill[i - 1, j] -1] = 1
                    if j - 1 >= 0:
                        if ill[i-1, j - 1] != 0:
                            adj[ill[i, j] - 1, ill[i-1, j - 1] - 1] = 1
                    if j + 1 <= 6:
                        if ill[i-1, j + 1] != 0:
                            adj[ill[i, j] - 1, ill[i-1, j + 1] - 1] = 1

                ## Check the (i+1, j) has the connection
                if i + 1 <= 5:
                    if ill[i + 1, j] != 0:
                        adj[ill[i, j] -1 , ill[i + 1, j] -1] = 1
                    if j - 1 >= 0:
                        if ill[i+1, j - 1] != 0:
                            adj[ill[i, j] - 1, ill[i+1, j - 1] - 1] = 1
                    if j + 1 <= 6:
                        if ill[i+1, j + 1] != 0:
                            adj[ill[i, j] - 1, ill[i+1, j + 1] - 1] = 1

                ## Check the (i, j-1) has the connection
                if j - 1 >= 0 :
                    if ill[i, j - 1] != 0:
                        adj[ill[i, j]-1, ill[i, j - 1]-1] = 1

                ## Check the (i, j+1) has the connection
                if j + 1 <= 6 :
                    if ill[i, j + 1] != 0:
                        adj[ill[i, j]-1, ill[i, j + 1]-1] = 1

    return adj