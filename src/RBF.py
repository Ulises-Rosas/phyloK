import numpy as np


def distance_matrix(a, b, M=None):
    """
    l2 norm squared matrix

    Time complexity:
    If M is full matrix, then O(np^2 + n^2p). Assuming p < n, then O(n^2p).
    If M is a diagonal or vector, then O(n^2p) regardless of the relationship between p and n.
    If M is None, then O(n^2p) regardless of the relationship between p and n.

    a: shape (m, p)
    b: shape (n, p)
    M: shape (p, p) or None
    
    returns: shape (m, n)
    """
    # a = X_test[:2, :]
    # b = X_train[:3, :]
    # # M = np.array([[2, 0], 
    # #               [0, 1]])
    # M = np.diag(M)

    # a = X_train
    # b = X_test
    # M = np.eye(a.shape[1])
    if M is None:
        # O(pn^2) for the whole matrix
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2
    
    else:
        # total time:
        # O(np + np + mnp + mn) = O(mnp) = O(n^2p) if m = n
        if len(M.shape) == 1:
            # O(np)
            A = ((a[ :, None, :] * M) @ a[:, :, None]).reshape(a.shape[0])
            # O(mp)
            B = ((b[ :, None, :] * M) @ b[:, :, None]).reshape(b.shape[0])
            # O(mpn + pn) = O(mnp)
            C = (a * M) @ b.T
            # O(mn)
            return np.add.outer(A,B) - 2*C


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


def nan_sqdist(a, b, use_t=False, T=None, k = 3):
    r"""
    Assuming a and b have 0 where it is missing.
    
    a**2 @ Mbf.T = [ a_1^2    [  |       |
                      ...       0/1 ... 0/1 
                     a_m^2 ]     |       | ]
                                 b1      bn <- presence/absence in b
                 = [ || a^{1}_1 ||^2  ...   || a^{n}_1 ||^2
                         
                     || a^{1}_m ||^2  ...   || a^{n}_m ||^2 ]
    In the sum, this activates l2 norm for overlapped features.
    The same applies to (b**2 @ Maf.T)

    a @ b.T = [ a_1^T b_1  ...  a_1^T b_n
                ...        ...      ...
                a_m^T b_1  ...  a_m^T b_n ]

    thus a**2 @ Mbf.T + (b**2 @ Maf.T).T - 2 * (a @ b.T):

    = [ ||a^{1}_1||^2 -2a_1^T b_1 +||b^{1}_1||^2  ...  ||a^{n}_1||^2 -2a_1^T b_n +||b^{1}_n||^2
                                                  ...
        ||a^{1}_m||^2 -2a_m^T b_1 +||b^{m}_1||^2  ...  ||a^{n}_m||^2 -2a_m^T b_n +||b^{m}_n||^2 ]
    

    O(n^2p) time complexity.
    """
    p = a.shape[1]

    Ma = ~np.isnan(a) # observed masks, shape (m, p), O(mp)
    Mb = ~np.isnan(b) # shape (n, p), O(np)
    a0 = np.where(Ma, a, 0.0)  # shape (m, p) zero-filled data, O(mp)
    b0 = np.where(Mb, b, 0.0)  # shape (n, p) zero-filled data, O(np)

    Maf = Ma.astype(a.dtype)
    Mbf = Mb.astype(b.dtype)

    n = Maf @ Mbf.T    # overlap counts, (m, n), O(mpn)
    # R^{m,p} R^{p,n} + (R^{n,p} R^{p,m}).T - 2 R^{m,p} R^{p,n} = R^{m,n}
    # O(mpn + mpn + mnp) = O(mnp) = O(n^2p) if m = n
    S = (a0**2) @ Mbf.T + ((b0**2) @ Maf.T).T - 2 * (a0 @ b0.T) 

    with np.errstate(divide="ignore", invalid="ignore"):
        D = S * (p / n)        # n == 0 -> inf or nan

    D[n == 0] = np.nan

    if sum(n == 0) > 0:
        k_linear_interpolation(D, k=k, T=T, use_t=use_t)

    return D

def k_linear_interpolation(D, k = 3, T = None, use_t = False):
    r"""
    D: distance matrix with NaN values, shape (m, n)
    k: number of nearest neighbors to use for interpolation
    T: distance matrix with NaN values, shape (m, n)
    use_t: if True, use T for finding nearest neighbors instead of D    


    O(n^2) algorithm 
    The algorithm iterates over each row of the distance matrix D, 
    and for each row, it finds the k nearest neighbors that have non-NaN values. 
    It then replaces the NaN values in that row with the mean of the k 
    nearest neighbors' distances.
    """
    # k = 2
    assert k > 0, "k must be a positive integer"

    if use_t:
        assert T is not None, "T must be provided when use_t is True"

    m,n = D.shape
    # O(n^2) for the whole matrix
    for i in range(m):
        # i = 0

        full_idx = ~np.isnan(D[i,:]) # O(n)
        if sum(full_idx) == n:
            continue

        # available neighbors
        full_idx[i] = False

        if sum(full_idx) == 0:
            # this means that there exist a single x_ik value at the row i
            # and there there no more values at column k.
            raise ValueError("All neighbors are NaN for row {}".format(i))

        if sum(full_idx) <= k:
            k_neighs_idx = full_idx
        else:
            # Get the k nearest neighbors, O(n)
            if use_t:
                k_neighs_idx = np.argpartition(T[i, full_idx], k)[:k]

            else:
                k_neighs_idx = np.argpartition(D[i, full_idx], k)[:k]

        # index with full values
        full_idx[i] = True

        D[i, ~full_idx] = np.mean(D[i, k_neighs_idx])

def RBF_kernel(a, b, gamma, M = None, nan_adjust = False, k = 3, T = None, use_t = False):
    """
    Radial Basis Function

    Time complexity:
    If M is full matrix, then O(np^2 + n^2p). Assuming p < n, then O(n^2p).
    If M is a diagonal or vector, then O(n^2p) regardless of the relationship between p and n.
    If M is None, then O(n^2p) regardless of the relationship between p and n.


    a: shape (m, p)\\
    b: shape (n, p)\\
    M: shape (p, p) or None\\
    nan_adjust: if True, then use nan_sqdist to compute the distance matrix, otherwise use distance_matrix\\
    k: number of nearest neighbors to use for interpolation if nan_adjust is True\\
    T: distance matrix with NaN values, shape (m, n)\\
    use_t: if True, use T for finding nearest neighbors instead of D\\


    returns: shape (m, n)
    """         
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    # data sim
    if nan_adjust:
        # O(n^2p) for the whole matrix and never uses M.
        D = nan_sqdist(a, b, use_t=use_t, T=T, k=k)

    else:
        D = distance_matrix(a, b, M=M)

    # D = np.where(D < 0, 0, D) # numerical stability
    tmp_rbf = -gamma * D

    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
    return tmp_rbf
