import numpy as np
# from numba import njit

def center_kernel(K):
    """
    Center the kernel matrix
    """

    K_row_mean = np.mean(K, axis = 1)
    K_col_mean = np.mean(K, axis = 0)
    K_all_mean = np.mean(K)

    K -= K_col_mean
    K -= K_row_mean.reshape(-1,1)
    K += K_all_mean

    return K



# @njit( nopython = True)
def mu_ber(f):
    f_tresh = 12
    f = np.clip(f, -f_tresh, f_tresh)
    mu =  1/(1 + np.exp(-f))
    return mu

# @njit( nopython = True)
def Q_ber(y, u, eps=1e-12, phi = 1):
    """
    Get the quasi-likelihood function for the logistic model
    """
    u = np.clip(u, eps, 1 - eps)
    return (1/phi) * (y * np.log(u) + (1 - y) * np.log(1 - u))

# @njit( nopython = True)
def Q(y, u, phi = 1):
    return (1/phi) * -0.5 * (y - u)**2

# @njit( nopython = True)
def Obj(y, f, p, dist = 'logistic', phi = 1):

    if len(f) == 2:
        f,p = f

    if dist == 'logistic':

        u = mu_ber(f)
        return - np.mean(Q_ber(y, u, phi = phi)) + p

    else:
        u = f
        return  - np.mean(Q(y, u, phi = phi))  + p
    
# @njit(nopython = True)
def backtracking_line_search(y, w_a, K_a, Kx, d, g_d, 
                             lam = 1, dist = 'logistic', phi = 1):
    """
    Also known as the Armijo's rule: It makes to satisfy 
    the first Wolfe condition. It is a sufficient decrease condition.
    """

    beta=1e-4
    p=0.5
    t = 10

    f_w_a = K_a @ w_a
    k_ad = K_a @ d

    w   =  w_a[1:] # penalized weights 
    d_w =  d[1:] # penalized weights direction

    # l/2 * (w_a + t*d)^T K_0 (w_a + t*d)
    #  = l/2 * (w + t*d_w)^T K_p (w + t*d_w)
    #  = l/2 * (  t^2 * d_w.T K_p d_w + 2t w.T K_p d_w + w.T K_p w )
    #  = l/2 * (  t^2 * c1 + 2t * c2  + c3)  
    #  PEnalized term

    c1 = d_w.T @ Kx @ d_w
    # c1 = d_w.T @ (k_ad - d_b)

    c2 = w.T @ Kx @ d_w
    # c2 = d_w.T @ (f_w_a - b) # d_w.T ( 1b + Kw - 1b)

    # since w.T K_p w is present on both sides of the inequality, 
    # it can be removed from the backtracking line search condition
    # this makes two changes:
    # 1. the PE term is now only the cross term and the quadratic term
    # 2. the obj term is now only the Q term
    
    PE = lambda x: (c1*(x**2) + 2*c2*x)*(lam/2)
    obj = Obj(y, f_w_a, 0, dist=dist, phi=phi)

    f_w_td = lambda x: f_w_a + x*k_ad

    while Obj(y, f_w_td(t), PE(t), dist=dist, phi=phi) > obj + (t * beta * g_d):
        t *= p
    return t



def newton_raphson_step(y, H, G, K_a = None, Kx = None, w_a = None, 
                        tol = 1e-5, dist = 'logistic', phi = 1, lam = 1):
    
    """
    returns lam_sq to check later for convergence
    """
    try:
        d = - np.linalg.solve( H, G ) # O(n^3)

    except np.linalg.LinAlgError:
        print("Linear algebra error occurred at solve. Adding hessian_tol to the diagonal of H and trying again.")
        d =  - np.linalg.solve( H + tol * np.eye(H.shape[0]), G ) # O(n^3)

    g_d = G.T @ d
    
    step = backtracking_line_search(y, w_a, K_a, Kx, d, g_d, 
                                    lam=lam, dist=dist, phi=phi)
    w_a += (step*d)

    return w_a, -g_d

def R_u(T, u, center = True):
    """
    Get the R centered matrix for the given T and u
    """
    # u = np.clip(u, -4, 4) # clip u to avoid numerical issues
    alpha = np.exp(-u)
    R_uc = np.exp( -alpha * T)
    return R_uc if not center else center_kernel(R_uc)


def dR_u(T, u, center = True):
    """
    Get the derivative of R centered matrix for the given T and u
    """
    alpha = np.exp(-u)
    dR_uc = alpha * T * np.exp( -alpha * T)
    return dR_uc if not center else center_kernel(dR_uc)



def backtracking_line_search_rij(y, f_w_a, w_a, u, T, Kx, d, g_d, 
                                 dist = 'logistic', phi = 1,
                                 reg_term = None, lam = 1,):
    """
    Also known as the Armijo's rule: It makes to satisfy 
    the first Wolfe condition. It is a sufficient decrease condition.
    """
    beta=1e-4
    p=0.5
    t=10

    w = w_a[1:] # penalized weights 
    b = w_a[0] # intercept

    if reg_term is None:
        obj = Obj(y, f_w_a, 0, dist=dist, phi=phi)
    else:
        obj = Obj(y, f_w_a, reg_term, dist=dist, phi=phi)


    def f_u_td(u_td):
        K_w_new = (Kx + R_u(T, u_td)) @ w
        if reg_term is None:
            return K_w_new + b
        else:
            return K_w_new + b, (lam/2)* (w.T @ K_w_new)


    # k = 0
    while Obj(y, f_u_td(u + t*d), 0, dist=dist, phi=phi) > obj + (t * beta * g_d):
        t *= p
        if t <= 1e-6:
            t = 1e-6
            break
        # k += 1
        # if k > 10:
        #     # print(t)
        #     break
    return t



def newton_raphson_step_rij(y, H, g, f_w_a, u, T, Kx, w_a, 
                        tol = 1e-5, dist = 'logistic', phi = 1,
                        reg_term = None, lam = 1):
    
    """
    returns lam_sq to check later for convergence
    """

    d = - g/H
    g_d = g * d

    step = backtracking_line_search_rij(y, f_w_a, w_a, u, T, Kx, d, g_d, 
                                 dist = dist, phi = phi,
                                 reg_term=reg_term, lam=lam)
    u += (step*d)
    u = np.clip(u, -4, 4) # clip u to avoid numerical issues
    return u, -g_d



# def centerK(K):
#     K_row_mean = np.mean(K, axis = 1)
#     K_col_mean = np.mean(K, axis = 0)
#     K_all_mean = np.mean(K)

#     K -= K_col_mean
#     K -= K_row_mean.reshape(-1,1)
#     K += K_all_mean

#     return K
# X = np.array(
#     [[1,2,3],
#      [3,2,1],
#      [2,1,3]], dtype=float)

# X = np.array(
#     [[1,2,3],
#      [3,2,1],
#      [2,1,3]], dtype=float)

# np.hstack((X[:,0], X[:,1]))
# np.exp(X)
# assert np.all(np.diag(X) == 0.0), 'OU needs path length distance'
# Y = np.random.random(X.shape)
# Du =  np.array([1,2,3])
# Y * Du

# Du.shape[0]



# C = np.eye(X.shape[0]) - np.ones(X.shape)/X.shape[0]
# C @ X + Y @ C

# np.round(centerK(X)+centerK(Y), 5) == np.round(C @ X @ C + C @ Y @ C, 5)



