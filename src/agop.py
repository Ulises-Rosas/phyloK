import numpy as np


def nabla_kw(l, K, X, M, gamma, wj):
    r"""
    Derivative of the kernel matrix with respect to the input data X.
    \partial k_{\ell, j} w_j / \partial x_{\ell} 

    total time complexity: O(pp + pn) <= O(pn + pn) = O(pn) if p < n.
    """
    # X = X_train
    # K = lkrr.K_train
    # M = np.eye(X.shape[1])
    # gamma = lkrr.gamma
    # wj = lkrr.alpha[1:]


    Dl = (X[l,:] - X).T #O(pn), R^{p,n}
    Dl_Kl = Dl * K[l,:] #O(pn), R^{p,n}
    Dl_kl_wj = Dl_Kl @ wj #O(pn), R^{p,1}

    if M is None:
        return -2 * gamma * Dl_kl_wj # O(p), R^{p,1}
    
    else:
        return -2 * gamma * M @ Dl_kl_wj # O(p^2), R^{p,1}

def agop(self, f_curr, y, W, lam, X, K, M, gamma):
    """
    Outer product gradient for recursive feature machines

    Algorithm exact time complexity: O(nmp^2 + pmn^2) <= O(pmn^2) if p < n, where n is the number of samples, m is the number of outputs, and p is the number of features.

    f_curr: current latent function values, shape (n, m)
    y: observed labels, shape (n, m)
    W: weights, shape (n, m)
    lam: regularization parameter
    X: input data, shape (n, p)
    K: kernel matrix, shape (n, n)
    M: prev agop matrix, shape (p, p)
    gamma: kernel parameter

    returns: M_new, shape (p, p)

    reference:
        Supplemntary material of the paper "Mechanism for feature learning in neural networks and backpropagation-free machine learning models"
    """

    mn = y.shape
    if len(mn) == 1:
        n,m = (mn[0], 1)
    else:
        n,m = mn

    Du, mu = self.get_Du_mu_internal(f_curr) # O(nm)
    e = y - mu # O(nm)
    Vu = self.get_Vu_func(f_curr) # O(nm)
    e /= Vu

    if m == 1:
        e  = e[:, np.newaxis]
        W  = W[:, np.newaxis]
        Du = Du[:, np.newaxis]

    c1 = m*n

    # O(nmp^2 + pmn^2)
    G = np.zeros((n, X.shape[1]))
    for l in range(n):
        nabla_f_l = 0
        # O(mp^2 + pnm)
        for j in range(m):
            d = (-1/c1) * Du[l,j] * e[l,j] + (lam/2) * W[l,j]
            # O(p^2 + pn)
            nabla_f_l += nabla_kw(l, K, X, M, gamma, W[:,j]) * d

        # print(nabla_f_l)
        G[l,:] = nabla_f_l

    G_mean = G.mean(axis=0)
    M_new = np.zeros((X.shape[1], X.shape[1]))
    # O(np^2)
    for l in range(n):
        # O(p)
        nabla_f_l = G[l,:] - G_mean
        # O(p^2)
        M_new += np.outer(nabla_f_l, nabla_f_l)

    return M_new / n

def distance_matrix(a, b, M=None):
    """
    l2 norm squared matrix

    Time complexity:
    when m = n and n > p: O(n^2p) 


    a: shape (m, p)
    b: shape (n, p)
    M: shape (p, p) or None
    
    returns: shape (m, n)
    """
    # a = X_test[:2, :]
    # b = X_train[:3, :]
    # M = np.array([[2, 0], 
    #               [0, 1]])

    # a = X_train
    # b = X_test
    # M = np.eye(a.shape[1])
    if M is None:
        # O(pn^2) for the whole matrix
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2
    
    else:
        # total time:
        # O(mn + mp^2 + np^2 + mpn + p^2n) 
        #  = O((m + n)p^2 + mpn)  
        # if m = n:
        # O(np^2 + n^2p) = O(npp + nnp) <= O(n^2p) if p < n

        # O(np^2)
        A = (a[ :, None, :] @ M @ a[:, :, None]).reshape(a.shape[0])
        # O(mp^2)
        B = (b[ :, None, :] @ M @ b[:, :, None]).reshape(b.shape[0])
        # O(mpn + p^2n)
        C = a @ M @ b.T
        # O(mn)
        return np.add.outer(A,B) - 2*C

def RBF_kernel(a, b, gamma, M = None):
    """
    Radial Basis Function

    Time complexity:
    when m = n and n > p: O(n^2p)

    """ 
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    # data sim
    tmp_rbf = -gamma * distance_matrix(a, b, M=M)

    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
    return tmp_rbf
