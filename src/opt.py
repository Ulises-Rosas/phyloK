import numpy as np
# from numba import njit
# from scipy.optimize import brent

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
    #  = l/2 * (  t^2 * c1            + 2t * c2        +    c3     )  
    #  PEnalized term

    c1 = d_w.T @ Kx @ d_w
    c2 = w.T @ Kx @ d_w

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
        if t <= 1e-12:
            t = 1e-12
            break
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

    return -g_d

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



def backtracking_line_search_rij(y, w_a, u, T, Kx, d, g_d, 
                                 dist = 'logistic',
                                 phi = 1, lam = 1,):
    """
    Also known as the Armijo's rule: It makes to satisfy 
    the first Wolfe condition. It is a sufficient decrease condition.
    """
    beta=1e-4
    p=0.5

    if d > 0:
        s = (4. - u)/d
    else:
        s = (-4. - u)/d

    t=min(10, s)


    w = w_a[1:] # penalized weights 
    b = w_a[0] # intercept

    def f_u_td(u_td):
        K_w_new = (Kx + R_u(T, u_td)) @ w
        return K_w_new + b, (lam/2)* (w.T @ K_w_new)

    obj = Obj(y, f_u_td(u), 0, dist=dist, phi=phi)

    # k = 0
    while Obj(y, f_u_td(u + t*d), 0, dist=dist, phi=phi) > obj + (t * beta * g_d):
        t *= p
        if t <= 1e-12:
            t = 1e-12
            break
        # k += 1
        # if k > 10:
        #     # print(t)
            # break
    return t


def backtracking_line_search_rij_gee(r, u, T, A_sq, d, g_d, 
                                     phi = 1,):
    """
    Also known as the Armijo's rule: It makes to satisfy 
    the first Wolfe condition. It is a sufficient decrease condition.
    """
    beta=1e-4
    p=0.5

    if d > 0:
        s = ( 4.0 - u)/d
    else:
        s = (-4.0 - u)/d

    t=min(10, s)

    def f(u):
        R = R_u(T=T, u=u, center=False)
        V = (A_sq[:, None] * R) * A_sq * phi
        return np.linalg.slogdet(V)[1] + r.T @ np.linalg.solve(V, r)

    obj = f(u)

    # k = 0
    while f(u + t*d) > obj + (t * beta * g_d):
        t *= p
        if t <= 1e-7:
            t = 0.0
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


def adaptive_delta(u_curr, g, s, rms_x, beta = 0.9, eps = 1e-8):
    """
    Adaptive delta method for updating u
    """
    rms_g = beta*s + (1 - beta)*g**2
    delta_x = - (np.sqrt( rms_x + eps )/np.sqrt(rms_g + eps)) * g
    
    rms_x = beta*rms_x + (1 - beta)*delta_x**2
    u_new = u_curr + delta_x
    return u_new,  s, rms_x

def adam(u_curr, k, g, v, s, lr = 0.001, gamma_v = 0.9, gamma_s = 0.999, eps = 1e-8):
    
    # biased decaying momentum
    v = gamma_v*v + (1 - gamma_v)*g
    # biased decaying sq. gradient
    s = gamma_s*s + (1 - gamma_s)*g**2
    # corrected decaying momentum
    v_hat = v/(1 - gamma_v**k)
    # corrected decaying sq. gradient
    s_hat = s/(1 - gamma_s**k)
    # new iterate
    u_new = u_curr - (lr*v_hat)/(np.sqrt(s_hat) + eps)

    return u_new, v, s


def nesterov_momentum(u_curr, v, nabla_f, beta = 0.9, lr = 0.001):
    # a,_ = nabla_f(u_curr + beta*v)
    # v = beta*v - lr*a

    v = beta*v - lr*nabla_f
    u_new = u_curr + v
    return u_new, v

def hyper_nesterov_momentum(u_curr, v, g_prev, g,
                            beta = 0.9, lr = 0.001, lrlr = 0.001):

    lr = lr  - lrlr*g*(-g_prev - beta*v)
    v = beta*v + g
    g_prev = g

    u_new = u_curr - lr*(g + beta*v)
    return u_new, v, g_prev, lr

def fibonacci_search(a, b, n, eps = 1e-3,
                     T = None, Kx = None, w = None,
                     y = None, lam_half = None, bias = None,
                     dist = 'logistic', phi = 1):


    def f(uc):
        R = R_u(T, np.exp(-uc), center = False)
        f_w = (Kx + R) @ w
        p = lam_half * (w.T @ f_w)
        return Obj(y, f_w + bias, p, dist = dist, phi = phi)

    gr = (np.sqrt(5) + 1) / 2
    s = (1-np.sqrt(5))/(1+np.sqrt(5))
    rho = (1 - s**n)/(gr *(1 - s**(n+1)) )

    d = rho*b + (1 - rho)*a
    yd = f(d)
    for i in range(1, n):
        # print(f"fibonacci search: a = {a}, b = {b}, f(a) = {f(a)}, f(b) = {f(b)}")

        if i == n - 1:
            c = eps*a + (1 - eps)*b
        else:
            c = rho*a + (1 - rho)*b

        yc = f(c)
        if yc < yd:
            b,d,yd = d,c,yc
        else:
            a,b = b,c

        rho = (1 - s**(n-i))/(gr *(1 - s**(n-i+1)) )


    return (a,b) if a < b else (b,a)


# def mybrent(a, b, n, eps = 1e-3,
#           T = None, Kx = None, w = None,
#           y = None, lam_half = None, bias = None,
#           dist = 'logistic', phi = 1):

#     def f(uc):
#         R = R_u(T, np.exp(-uc), center = False)
#         f_w = (Kx + R) @ w
#         p = lam_half * (w.T @ f_w)
#         return Obj(y, f_w + bias, p, dist = dist, phi = phi)


#     res = brent(f, brack=(a, b), tol=eps, maxiter = n)
#     return res



            # adaptive gradient
            # self.grad_s += g**2
            # u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g

            # RMSprop
            # self.grad_s = self.beta*self.grad_s + (1 - self.beta)*g**2
            # u_new = u_curr - (mylr/np.sqrt(self.grad_s + self.grad_eps)) * g

            # adaptive delta
            # u_new, rms_x, self.grad_s = adaptive_delta(u_curr, g, self.grad_s, rms_x, self.beta, self.grad_eps)

            # ADAM optimizer
            # u_new, v, s = adam(u_curr, i + 1, g, v, s, mylr)

            # nesterov momentum
            # nabla_f,_ = grad_wrt_u(u_curr - self.beta * v)
            # u_new, v = nesterov_momentum(u_curr, v, nabla_f, self.beta, mylr)

            # hypergradient nesterov momentum
                # u_new, v, g_prev, mylr = hyper_nesterov_momentum(u_curr, v, g_prev, g, self.beta, mylr, mylrlr)




