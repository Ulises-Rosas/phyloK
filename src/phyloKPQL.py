import sys
import time


import numpy as np
from phylokrr.kernels import linear_kernel
# from scipy import linalg as la
from numpy import linalg as la

from phylokrr_dev.opt import newton_raphson_step, newton_raphson_step_rij, R_u, dR_u, backtracking_line_search_rij

def distance_matrix(a, b):
    """
    l2 norm squared matrix
    """
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2


def RBF_kernel(a, b, gamma, extra_K = None):
    """
    Radial Basis Function
    """ 
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    # data sim
    tmp_rbf = -gamma * distance_matrix(a, b)

    # if extra_K is not None:
    #     tmp_rbf += extra_K

    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation

    if extra_K is not None:
        tmp_rbf += extra_K

    return tmp_rbf


class KRGEE:
    """
    Kernel Ridge Generalized Estimating Equations for 1 cluster.
    Using 1 cluster is what phylogenetic regresion deals with.
    """
    def __init__(self, 
                 kernel='rbf',
                 max_iter=300, 
                 lam=0.1,
                 warm_start = False,
                 tol = 0.001,
                 verbose = False,
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 add_extraK= False
                 ) -> None:

        assert max_iter >= 1, "number of iterations should be a least 1" 
        
        self.add_extraK = add_extraK
        self.reg = 'continous'
        self.process = "BM"
        self.inv_calculated = False
        
        self.Rnn = None

        self.alpha = np.array([])
        self.psc = 0 # phylo signal constant

        self.verbose = verbose
        self.warm_start = warm_start

        self.kernel = kernel
        self.lam = lam
        
        self.gamma = 0.1
        self.c = 0.1

        self.max_iter = max_iter
        self.tol = tol

        # speicify if the correlation matrix is needed
        # it might be the case that the input matrix is already
        # in form of a correlation matrix
        self.get_corr = get_corr 

        self.Du_mu_func = lambda x: (np.ones(x.shape[0]), x)
        self.Vu_func = lambda x: np.ones( x.shape[0] )
        
        self.X = np.array([])
        self.y = np.array([])

        self.copy_R_inv = copy_R_inv
        self.R_inv = np.array([])
        
        self.copy_K = copy_K_train
        self.K_train = np.array([])
        self.K_col_mean = np.array([])
        self.K_all_mean = np.array([])


        self.phi = 1
        self.fix_phi = fix_phi
        self.r = np.array([]) # residuals
        self.dof = 1 # degrees of freedom
        self.L_inv = np.array([]) # R^-1/2


        self.f_curr = np.array([]) # current function values

        self._verbose = False
        self.grad_tol = 1e-10
        self.hessian_tol = 1e-10
        self.grad_clip = 2 # gradient clipping

        self.grad_s = 0.0 # s 
        self.grad_eps = 1e-8
        self.lr = 0.001 # learning rate
        self.beta = 0.9 # beta for rmsprop

    def __repr__(self):
        return f"KRGEE(kernel={self.kernel}, max_iter={self.max_iter}, lam={self.lam}, warm_start={self.warm_start}, tol={self.tol})"
    
    
    def center_kernel(self, K, trainxtest = False, save_means = True):
        """
        Center the kernel matrix

        ref: Kernel methods in computational biology, 
        Scholkopf, Tsuda, Vert. pp. 50\\
        \\tilde K = (I - e/n)( K )(I - e/n)

        e is singular matrix of ones\\
        I is identity matrix\\
        n is the number of samples\\
        K is the kernel matrix of size n x p

        Only works for square matrices 

        Here is a general attempt for non-square matrices
        """
        if K is None:
            n = self.X.shape[0]
            return np.zeros((n,n))

        K_row_mean = np.mean(K, axis = 1)

        if not (save_means or trainxtest):
            K_col_mean = np.mean(K, axis = 0)
            K_all_mean = np.mean(K)

            K -= K_col_mean
            K -= K_row_mean.reshape(-1,1)
            K += K_all_mean
            return K

        if not trainxtest:
            self.K_col_mean = np.mean(K, axis = 0)
            self.K_all_mean = np.mean(K)

        assert len(self.K_col_mean) > 0, 'col means from traning data missing'
        K -= self.K_col_mean
        K -= K_row_mean.reshape(-1,1)
        K += self.K_all_mean
        return K

    def set_params(self, **params):
        """
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not recognized.")

    def get_params(self):
        return  {'max_iter': self.max_iter, 
                'lam': self.lam, 
                'intercept': self.alpha[0] if len(self.alpha) > 0 else None, 
                'tol': self.tol}

    def get_K(self, A, B, R = None, trainxtest=False, center = True):
        """
        Get the kernel matrix
        depending on the kernel type specified
        a parameter is passed to the kernel function
        Otherwise, the input matrix A is returned

        Parameters
        ----------
        A : np.ndarray
            input matrix
        B : np.ndarray
            input matrix

        Returns
        -------
        np.ndarray
            kernel
        """
        if self.kernel == 'rbf':

            K = RBF_kernel(A, B, self.gamma)

        elif self.kernel == 'linear':
            K =  linear_kernel(A, B, self.c)

        else:
            K =  A

        if R is not None:
            K += R

        return self.center_kernel(K, trainxtest) if center else K
    
    def pre_process_V(self, V):
        """
        Pre-process the V matrix
        1. Normalize the matrix
        2. Get the n-th root of the determinant
        3. Normalize the matrix again

        it improves the numerical stability of the algorithm

        refs: 

        https://daijiang.github.io/phyr/reference/pglmm_compare.html
        https://ives.labs.wisc.edu/pdf/correlateddata.pdf (pp. 94)

        """
        V = np.asfortranarray(V, dtype=V.dtype)

        n, _ = V.shape
        V = V/np.max(V)
        detV_n = np.linalg.det(V)**(1/n)
        V = V/detV_n if detV_n != 0 else V
        
        return V

    def get_R(self, vcv):
        """
        Get correlation matrix from vcv matrix.

        Raises
        ------
        ValueError
            If the covariance/variance matrix has non-positive diagonal values.
        """
        vcv = np.asarray(vcv, dtype=float)
        if vcv.ndim != 2 or vcv.shape[0] != vcv.shape[1]:
            raise ValueError("vcv must be a square matrix")

        diag = np.diag(vcv)
        if np.any(~np.isfinite(diag)) or np.any(diag <= 0):
            raise ValueError("vcv must have strictly positive, finite diagonal entries")

        kr = 1 / np.sqrt(diag)
        R = (kr[:, None] * vcv) * kr
        return R
    
    def _invR(self, R):
        """
        Inverse of the correlation matrix
        """
        # L = np.linalg.cholesky(R + np.eye(R.shape[0]) * self.grad_tol)
        # self.L_inv = np.linalg.inv(L)

        # return self.L_inv.T @ self.L_inv

        try:
            E,Q = la.eigh(R)
        except la.LinAlgError:
            E,Q = la.eigh(R + np.eye(R.shape[0]) * self.hessian_tol)

        # E,Q = la.eigh(R)
        self.L_inv = Q @ np.diag(E**(-1/2)) @ Q.T
        return Q @ np.diag(E**(-1)) @ Q.T
    
    def get_R_inv(self, vcv):
        """
        Get correlation matrix from vcv matrix
        and return its inverse
        """
        # TODO: get_corr might not be needed
        # as if it is already a correlation matrix
        # is it is like divding by 1
        # vcv = self.pre_process_V(vcv)

        R = self.get_R(vcv) if self.get_corr else vcv

        return self._invR(R)
        
    def set_R_inv(self, vcv):
        """
        Set correlation matrix from vcv matrix
        and return its inverse

        The general logic is as follows:

        if copy_S and S == 0:
            set_S

        elif not copy_S and S == 0:
            set_S (overwrite)

        elif copy_S and S != 0:
            do nothing

        elif not copy_S and S != 0:
            set_S (overwrite)        

        where S is Some matrix,
        in this case the correlation matrix

        """
        if self.copy_R_inv and len(self.R_inv) != 0:
            pass

        else:
            if isinstance(vcv, np.ndarray):
                self.R_inv = self.get_R_inv(vcv)
                self.inv_calculated  = True

            else:
                n,_ = self.X.shape
                self.R_inv = np.eye(n)
                self.L_inv = np.eye(n)

    def set_K_train(self, X):
        """
        Set the kernel matrix

        The general logic is as follows:

        if copy_S and S == 0:
            set_S

        elif not copy_S and S == 0:
            set_S (overwrite)

        elif copy_S and S != 0:
            do nothing

        elif not copy_S and S != 0:
            set_S (overwrite)        

        where S is Some matrix,
        in this case the kernel matrix

        Parameters
        ----------
        X : np.ndarray
            input matrix
        """

        if X is None:
            self.K_train = None
            return 

        if self.copy_K and len(self.K_train) != 0:
            pass

        else:
            Kc = self.get_K(X, X, R = self.Rnn)
            self.K_train = np.asfortranarray(Kc, dtype=float)


    def get_Du(self, f_curr):
        """
        Get mu vector (estimator) and W matrix with mu derivatives 
        with respect to f(x).


        Wii = \\frac{\partial \mu}{\partial f(x)}
        if \mu is the identity function, then Wii = 1


        from the current function values.

        The function is will depend on the model being used.

        Parameters
        ----------
        f_curr : np.ndarray
            current function values

        Returns
        -------
        np.ndarray, np.ndarray
            Du, u
        """
        return self.Du_mu_func(f_curr)
    
    def get_Vu_sqrt_inv(self, f_curr):
        """
        get the square root of the inverse of 
        the variance of the estimator mu.

        the function is model dependent

        Parameters
        ----------
        f_curr : np.ndarray
            current function values

        Returns
        -------
        np.ndarray
            square root of the inverse of the variance
        """
        # Vu = self.Vu_func(f_curr)
        # Vu[Vu < self.epsilon] = self.epsilon
        # return Vu**(-1/2)
        return self.Vu_func(f_curr)**(-1/2)

    def set_df(self, H):
        """
        residual degrees of freedom

        ref: Elements of Statistical Learning, Hastie, Tibshirani, Friedman
        pp. 233

        Parameters
        ----------
        H : np.ndarray
            Hessian matrix

        lam : float
            regularization parameter

        """
        n = H.shape[0]

        theta = np.linalg.eigvalsh(H) # O(n^3)
        theta = np.real(theta)
        df_model = np.sum(theta / (theta + self.lam))
        self.dof = np.clip(n - df_model, self.grad_tol, n)

    def set_rdf(self, y, y_hat):
        """
        residual degrees of freedom
        using the covariance method

        ref: Elements of Statistical Learning, Hastie, Tibshirani, Friedman
        pp. 77, 233
        """
        # effective number of parameters
        df_model = np.cov(y_hat, y, ddof=1)[0, 1] / np.var(y - y_hat, ddof=1)
        n = len(y_hat)
        self.dof = np.clip(n - df_model, self.grad_tol, n)

    def set_phi(self, e):
        # pearson residual
        err = e * self.Vu_func(self.f_curr)**(-1/2)
        self.phi = np.dot(err, err)/self.dof # O(n)
    
    def get_V_Inv(self, f_curr):
        """
        Get the square root of the inverse of the V matrix
        V = A^1/2 R A^1/2 \phi
        A = variance of \mu
        R = correlation matrix
        phi = dispersion parameter

        V^(-1) = ( A^1/2  R      A^1/2  \phi)^-1
               =   A^-1/2 R^-1   A^-1/2 \phi^-1

        O(n^2) complexity

        Parameters
        ----------
        f_curr : np.ndarray
            current function values
        Returns
        -------
        np.ndarray
            V^(-1/2)
        """
        A_sqInv = self.Vu_func(f_curr)**(-1/2)
        V_Inv  = (A_sqInv[:, None] * self.R_inv) * A_sqInv
        V_Inv *= (self.phi)**(-1) # O(n^2)    
        return V_Inv

    def krgee_fit(self, K, y):
        
        n = y.shape[0]
        p = K.shape[1] if K is not None else 0
        K_dtype = K.dtype if K is not None else float

        ones = np.ones(n, dtype=K_dtype)
        
        if p > 0:
            K_0 = np.zeros((n + 1, p + 1), dtype=K_dtype)
            K_0[1:, 1:] = K
            lK_0 = self.lam * K_0

        else:
            lK_0 = np.array([[0]], dtype=K_dtype)


        # concatenate ones to K for intercept
        K_a = np.column_stack((ones, K)) if p > 0 else ones.reshape(-1, 1)
        K_a = np.asfortranarray(K_a, dtype=K_dtype)

        if not self.warm_start or np.any(np.isnan(self.alpha)):
            self.alpha = np.zeros(p + 1) # alpha weights
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K_dtype)
            self.phi = 1

        assert len(self.alpha) > 0, "The model needs to be fitted first"
    
        f_curr = K_a @ self.alpha
        conv = False
        for i in range(self.max_iter):
            # get derivatives of \mu and \mu
            
            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr) # O(n^2)
    
            DuDa = Du[:,None] * K_a # O(n^2)
            DuDaT_V = np.matmul(DuDa.T, V_Inv) # O(n^3)
            DuDaT_V_DuDa = np.matmul(DuDaT_V, DuDa) # O(n^3)

            H = ((1/n)*(DuDaT_V_DuDa)) + lK_0 # O(n^2)
            # if np.linalg.cond(H) > 1/sys.float_info.epsilon:
            #    H += np.eye(H.shape[0]) * self.hessian_tol

            e = y - u # O(n)

            # gradient
            G =  ((-1/n)*(DuDaT_V @ e)) + lK_0 @ self.alpha # O(n^2)

            G_norm = np.linalg.norm(G, ord=2)
            # gradient clipping, O(n)
            # to avoid exploding gradients
            # if G_norm >= self.grad_clip:
            #     G *= (self.grad_clip / G_norm)
            if G_norm/max(1, np.linalg.norm(self.alpha, ord=2)) <= 1e-8:
                if self.verbose:
                    print("Gradient norm is below the clipping threshold")
                conv = True
                break

            try:
                self.alpha -= np.linalg.solve( H, G ) # O(n^3)

            except np.linalg.LinAlgError:
                print("Linear algebra error occurred at solve. Adding hessian_tol to the diagonal of H and trying again.")
                self.alpha -= np.linalg.solve( H + self.hessian_tol * np.eye(H.shape[0]), G ) # O(n^3)
    
            # new optimal function val
            f_new = K_a @ self.alpha
            if self.check_convergence(f_new, f_curr) and i > 1:
                if self.verbose:
                    print("Converged at update = %i" %i)
                conv = True
                break

            if self._verbose:
                print("Iteration: ", i)
                print('phi: ', self.phi)
                print('fix_phi: ', self.fix_phi)
                print()
        
            if not self.fix_phi:
                if p > 0:
                    #  hessian of the penalized coefficients (alpha without intercept)
                    HA = H[1:,1:]
                    # HA = ((1/n)*(DuDaT_V_DuDa[1:, 1:])) + lK # O(n^2)
                    # UHA = (DuDaT_V_DuDa[1:, 1:] + lK) # O(n^2)
                    self.set_df(HA) # O(n^3)
                else:
                    self.dof = n - 1
                
                self.set_phi(e)

            f_curr = f_new

        self.f_curr = f_curr
        self.y = y.copy()

        if not conv:
            print("No convergence after %i iterations" % self.max_iter)
            print("gamma, phi, lam:")
            print(self.gamma, self.phi, self.lam)


    def dR_R(self, u_curr, T):
        """
        Get the derivative of the correlation matrix with respect to u
        and the correlation matrix itself. Both are centered.

        """
        alpha_u = np.exp(-u_curr)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        dR = self.center_kernel(dR_uc, save_means=False)
        R = self.center_kernel(R_uc, save_means=False)
        return dR, R

    def opt_rij_alpha(self, y, T, Kx, uc = 0.0):
        """

        u used for phylogenetic correlation alpha = exp(-u), 
        dont get confuse with u from best average estimator u = mu(f_curr)        
        """

        n = y.shape[0]
        w = self.alpha[1:]

        u_curr = uc
        conv = False
        for i in range(self.max_iter):
            # get derivatives of \mu and \mu
            dR, R = self.dR_R(u_curr, T)
            # print("dR, R: ", dR, R)

            f_curr = (Kx + R) @ w + self.alpha[0]
            # print("f_curr:", f_curr)

            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr) # O(n^2)
    
            DuDa =  (Du[:,None] * dR) @ w
            DuDaT_V = np.matmul(DuDa.T, V_Inv) # O(n^3)
            DuDaT_V_DuDa = np.matmul(DuDaT_V, DuDa) # O(n^3)

            H = (1/n)*(DuDaT_V_DuDa) # Hessian is an scalar
            # clip the hessian to avoid numerical issues
            # if H == np.inf or H == 0:
            #     print("dR: ", dR)
            #     print('alpha: ', self.alpha)
            #     print("R: ", R)
            #     print("R_uc: ", np.exp(-np.exp(-u_curr) * T))
            #     print("f_curr: ", f_curr)
            #     print("Du: ", Du)
            #     print("u: ", u)
            #     print("V_Inv: ", V_Inv)
            #     print("DuDa: ", DuDa)
            #     print("DuDaT_V: ", DuDaT_V)
            #     print("DuDaT_V_DuDa: ", DuDaT_V_DuDa)
            #     raise ValueError("Hessian is zero or infinity. Check the input data and parameters.")
    
            H = np.clip(H, -self.grad_clip, self.grad_clip)



            # print("Hessian from alpha param: ", H)

            # gradient
            g = -(1/n)*(DuDaT_V @ (y - u))
            g = np.clip(g, -self.grad_clip, self.grad_clip)
            # print("Gradient from alpha param: ", g)

            u_new, lam_sq = newton_raphson_step_rij(y, H, g, f_curr,
                                                    u_curr, T, Kx, 
                                                    self.alpha, 
                                                    self.hessian_tol, 
                                                    dist = self.reg, 
                                                    phi = self.phi)
            # print("u_new, u_curr: ", u_new, u_curr)
            u_new = np.clip(u_new, -4, 4)         
            if self.check_convergence(u_new, u_curr, 
                                    #   g_norm = np.linalg.norm(g, ord=2), 
                                      g_norm = np.abs(g), 
                                      lam2 = lam_sq, 
                                      i = i):
                conv = True
                self.psc = u_new
                break

            u_curr = u_new

        if not conv:
            print("No convergence after %i iterations" % self.max_iter)
            print(f"curr. phylo signal: {np.exp(-u_curr)}")

        self.psc = u_curr

    def opt_rij_gradient_descent(self, y, T, Kx, uc = 1):
        """
        Optimize the phylogenetic correlation parameter using gradient descent
        """
        n = y.shape[0]
        w = self.alpha[1:]

        u_curr = uc
        conv = False
        self.grad_s = 0.0
        for i in range(self.max_iter):
            dR, R = self.dR_R(u_curr, T)
            f_curr = (Kx + R) @ w + self.alpha[0]

            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr)

            DuDa =  (Du[:,None] * dR) @ w
            DuDaT_V = np.matmul(DuDa.T, V_Inv)

            g = -(1/n)*(DuDaT_V @ (y - u))

            # update u using a simple gradient descent step
            # step_size = 0.001  # You may want to tune this parameter
            # u_new = u_curr - step_size * g

            # self.grad_s = self.beta*self.grad_s + (1 - self.beta)*g**2
            # u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g

            u_new, lam_sq = newton_raphson_step_rij(y, 1, g, f_curr,
                                                    u_curr, T, Kx,
                                                    self.alpha,
                                                    self.hessian_tol,
                                                    dist=self.reg,
                                                    phi=self.phi)

            u_new = np.clip(u_new, -4, 4)  # Clip u_new to avoid extreme values
            if self.check_convergence(u_new, u_curr, 
                                      g_norm = np.abs(g), 
                                      lam2 = lam_sq, 
                                    #   lam2 = self.tol*2, 
                                      i = i):
                conv = True
                self.psc = u_new
                break

            u_curr = u_new

        if not conv:
            print("No convergence after %i iterations" % self.max_iter)
            print(f"curr. phylo signal: {np.exp(-u_curr)}")

        self.psc = u_curr

    def opt_rij_gd_fullK(self, y, T, Kx, uc = 1):
        """
        Optimize the phylogenetic correlation parameter using gradient descent
        """
        n = y.shape[0]
        w = self.alpha[1:]

        u_curr = uc
        self.grad_s = 0.0

        conv = False
        for i in range(self.max_iter):
            dR, R  = self.dR_R(u_curr, T)
            f_curr = (Kx + R) @ w + self.alpha[0]

            Du, u   = self.get_Du(f_curr)
            V_Inv   = self.get_V_Inv(f_curr)
            DuDa    = (Du[:,None] * dR) @ w
            DuDaT_V = np.matmul(DuDa.T, V_Inv)

            reg_term = (self.lam/2) * (w.T @ dR @ w)
            g = -(1/n)*(DuDaT_V @ (y - u)) + reg_term
            # g = np.clip(g, -self.grad_clip, self.grad_clip)

            # update u using gradient descent
            # u_new = u_curr - self.lr * g

            # adaptive gradient
            # self.grad_s += g**2
            # u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g

            # RMSprop
            self.grad_s = self.beta*self.grad_s + (1 - self.beta)*g**2
            u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g
    
            # d = - g
            # g_d = g * d
            # lam_sq = -g_d
            # step = backtracking_line_search_rij(
            #     y, f_curr, self.alpha, u_curr, T, Kx, d, g_d, 
            #     dist = self.reg, phi = self.phi,
            #     reg_term=reg_term, lam=self.lam
            # )
            # u_new = u_curr + (step*d)

            u_new = np.clip(u_new, -4, 4)  # Clip u_new to avoid extreme values
            if self.check_convergence(u_new, u_curr, 
                                      g_norm = np.abs(g), 
                                    #   lam2 = lam_sq,
                                      lam2 = 2*self.tol,
                                      i = i):
                # print(f"converge for r (iter. {i})")
                conv = True
                self.psc = u_new
                break

            u_curr = u_new

        if not conv:
            print("No convergence after %i iterations" % self.max_iter)
            print(f"curr. phylo signal: {np.exp(-u_curr)}")

        self.psc = u_curr


    def opt_weights(self, K, y, Kx = None):
        
        n = y.shape[0]
        p = K.shape[1] if K is not None else 0
        K_dtype = K.dtype if K is not None else float

        ones = np.ones(n, dtype=K_dtype)
        
        if p > 0:
            K_0 = np.zeros((n + 1, p + 1), dtype=K_dtype)
            K_x = K if Kx is None else Kx
            K_0[1:, 1:] = K_x
            lK_0 = self.lam * K_0

        else:
            lK_0 = np.array([[0]], dtype=K_dtype)


        # concatenate ones to K for intercept
        K_a = np.column_stack((ones, K)) if p > 0 else ones.reshape(-1, 1)
        K_a = np.asfortranarray(K_a, dtype=K_dtype)

        if not self.warm_start or np.any(np.isnan(self.alpha)):
            self.alpha = np.zeros(p + 1) # alpha weights
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K_dtype)
            self.phi = 1

        assert len(self.alpha) > 0, "The model needs to be fitted first"
    
        f_curr = K_a @ self.alpha
        conv = False
        for i in range(self.max_iter):
            # get derivatives of \mu and \mu
            
            Du, u = self.get_Du(f_curr)
            V_Inv = self.get_V_Inv(f_curr) # O(n^2)
    
            DuDa = Du[:,None] * K_a # O(n^2)
            DuDaT_V = np.matmul(DuDa.T, V_Inv) # O(n^3)
            DuDaT_V_DuDa = np.matmul(DuDaT_V, DuDa) # O(n^3)

            H = (1/n)*(DuDaT_V_DuDa) + lK_0 # O(n^2)
            self.r = y - u # O(n)

            # gradient
            G =  (-1/n)*(DuDaT_V @ self.r) + lK_0 @ self.alpha # O(n^2)

            self.alpha, lam_sq = newton_raphson_step(
                y, H, G, K_a, K_x, self.alpha, self.hessian_tol,
                dist=self.reg, phi = self.phi, lam=self.lam
            )
            
            # self.alpha += d
    
            # new optimal function val
            f_new = K_a @ self.alpha
            if self.check_convergence(f_new, f_curr, 
                                      g_norm = np.linalg.norm(G, ord=2), 
                                      lam2 = lam_sq, 
                                      i = i):
                conv = True
                f_curr = f_new
                break

            f_curr = f_new

            if self._verbose:
                print("Iteration: ", i)
                print('phi: ', self.phi)
                print('fix_phi: ', self.fix_phi)
                print()
        
            if not self.fix_phi:
                if p > 0:
                    #  hessian of the penalized coefficients (alpha without intercept)
                    HA = H[1:,1:]
                    self.set_df(HA) # O(n^3)
                else:
                    self.dof = n - 1
                
                self.set_phi(e)

        self.f_curr = f_curr
        self.y = y.copy()

        if not conv:
            print("No convergence after %i iterations" % self.max_iter)
            print("gamma, phi, lam:")
            print(self.gamma, self.phi, self.lam)

    def mu(self, f):
        return f

    def Q(self, y, u):
        return (1/self.phi) * -0.5 * (y - u)**2

    def Obj(self, y, f, p):

        u = self.mu(f)
        return - np.mean(self.Q(y, u)) + p

    def fit0(self, X, y, vcv = None):
        """
        Fit the model
        """
        self.X = X # store the training data for prediction
        self.set_R_inv(vcv) # it returns identity matrix if vcv is None
        self.set_K_train(self.X)

        # self.krgee_fit(self.K_train, y)
        self.opt_weights(self.K_train, y)

    def update_K_train(self, u_new, T, K_x_uc):
        """
        Update uncentered Rnn and K_train
        """
        self.Rnn = R_u(T, u_new, center = False)
        # print("Rnn at update K_train")
        # print(self.Rnn[0:3, 0:3])

        self.K_train = self.center_kernel(K_x_uc + self.Rnn, save_means = True)

    def get_new_score(self, K_x, y):
        """
        it uses self.K_train previously computed with the old u value

        Get the new score after updating the phylogenetic correlation parameter
        and update self.f_curr accordingly

        K_x: if it is None, then self.K_train is used for penalization term
        """
        f_wa = self.K_train @ self.alpha[1:]
        f_curr = f_wa + self.alpha[0]

        if K_x is not None:
            p = (self.lam/2) * (self.alpha[1:].T @ K_x @ self.alpha[1:])
        else:
            p = (self.lam/2) * (self.alpha[1:].T @ f_wa)

        return self.Obj(y, f_curr, p = p)

    def decoupleR(self, y, X, vcv):
        self.Rnn = R_u(vcv, self.psc, center = False)

        K_x_uc = self.get_K(X, X, center = False)
        K_x = self.center_kernel(K_x_uc, save_means = False)

        self.K_train = self.center_kernel(K_x_uc + self.Rnn, save_means = True)
        self.opt_weights(self.K_train, y, Kx = K_x)
        self.opt_rij_gradient_descent(y, T = vcv, Kx = K_x, uc = self.psc)
        # self.opt_rij_alpha(y, T = vcv, Kx = K_x, uc = self.psc)

        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)

        curr_score = self.get_new_score(K_x, y)

        # let re-use alpha weights from the previous 
        # step to speed up convergence
        # alt_iter = 0
        for i in range(self.max_iter):
            self.opt_weights(self.K_train, y, Kx=K_x) # update weights
            self.opt_rij_gradient_descent(y, T=vcv, Kx=K_x, uc=self.psc)
            # self.opt_rij_alpha(y, T = vcv, Kx = K_x, uc = self.psc)

            self.update_K_train(u_new=self.psc, T=vcv, K_x_uc=K_x_uc)
            new_score = self.get_new_score(K_x, y)

            lam_sq = curr_score - new_score
            if lam_sq/2 < self.tol:
                print(f"Alternate iteration {i + 1}: new score = {new_score}, old score = {curr_score}")
                print()
                break

            if new_score < -4*y.shape[0]:
                # since it is a minimization problem, a very low score 
                # indicates that the model is not converging. However
                print("Warning: new score is already very well optimized")
                break

            curr_score = new_score

    def fullK_opt(self, y, X, vcv):
        self.Rnn = R_u(vcv, self.psc, center = False)

        K_x_uc = self.get_K(X, X, center = False)
        K_x = self.center_kernel(K_x_uc, save_means = False)

        self.K_train = self.center_kernel(K_x_uc + self.Rnn, save_means = True)

        self.opt_weights(self.K_train, y, Kx = None)
        # print("0. psc before opt:", self.psc)
        self.opt_rij_gd_fullK(y, T = vcv, Kx = K_x, uc = self.psc)
        # print("0. psc after opt:", self.psc)

        self.update_K_train(u_new = self.psc, T = vcv, K_x_uc = K_x_uc)

        curr_score = self.get_new_score(None, y)
        curr_vals = np.hstack((self.f_curr, self.psc))


        # self.warm_start = True
        conv = False
        for i in range(self.max_iter):
            self.opt_weights(self.K_train, y, Kx=None) # update weights
            # print(f"{i + 1}. psc before opt:", self.psc)
            self.opt_rij_gd_fullK(y, T=vcv, Kx=K_x, uc=self.psc)
            # print(f"{i + 1}. psc after opt:", self.psc)

            self.update_K_train(u_new=self.psc, T=vcv, K_x_uc=K_x_uc)

            new_score = self.get_new_score(None, y)
            new_vals = np.hstack((self.f_curr, self.psc))

            lam_sq = curr_score - new_score
            if self.check_convergence(new_vals, curr_vals, 
                                      g_norm=self.grad_tol*2, 
                                      lam2=lam_sq, 
                                      i = i + 1):
            # if lam_sq/2  < self.tol:
#            # if alt_iter > 3:
                # print(f"Converged at iteration {i + 1}, phylogenetic correlation parameter: {self.psc}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")
                # print("converged.")
                # print()
                conv = True
                break
            curr_score = new_score

        if not conv:
            print(f"Din't converged. phylogenetic correlation parameter: {self.psc}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")

        # self.warm_start = False

    def dR_R_ls(self, u_curr, T):
        """
        Get the derivative of the correlation matrix with respect to u
        and the correlation matrix itself. Both are centered.

        """
        alpha_u = np.exp(-u_curr)
        R_uc = np.exp( -alpha_u * T)
        dR_uc = alpha_u * T * R_uc
        return dR_uc, R_uc

    def opt_R_ls(self, u, T):

        u_curr = u
        
        dR, R = self.dR_R_ls(u_curr, T) # double computation for R. Fix this later
        A_sq = self.Vu_func(self.f_curr)**(1/2)
        V = (A_sq[:, None] * R) * A_sq * self.phi
        curr_obj = np.linalg.slogdet(V)[1] + self.r.T @ np.linalg.solve(V, self.r)
        # print("curr_obj: ", curr_obj)
        
        self.grad_s = 0.0
        for i in range(self.max_iter):
            g = np.trace(np.linalg.solve(V, dR)) - self.r.T @ np.linalg.solve(V, dR @ np.linalg.solve(V, self.r))
            
            u_new = u_curr - self.lr * g
            
            # adaptive gradient
            # self.grad_s += g**2
            # u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g

            # self.grad_s = self.beta*self.grad_s + (1 - self.beta)*g**2
            # u_new = u_curr - (self.lr/np.sqrt(self.grad_s + self.grad_eps)) * g

            u_new = np.clip(u_new, -4, 4)
            dR_new, R_new = self.dR_R_ls(u_new, T)
            V_new = (A_sq[:, None] * R_new) * A_sq * self.phi
            new_obj = np.linalg.slogdet(V_new)[1] + self.r.T @ np.linalg.solve(V_new, self.r)

            if (curr_obj - new_obj)/2 < self.tol:
                break

            u_curr = u_new
            curr_obj = new_obj
            dR = dR_new
            R = R_new
        # print(f"Final phylogenetic correlation parameter: {u_curr}")
        return u_curr, R


    def fit(self, X, y, vcv = None):
        """
        Fit the model
        """
        self.psc = 0.0 # arbitrary initial value for the phylogenetic correlation parameter
        self.X = X # store the training data for prediction

        if self.add_extraK:
            self.set_R_inv(None) # it returns identity matrix if vcv is None

            if self.process == 'BM':
                assert np.all(np.diag(vcv) != 0.0),\
                        "BM needs a correlation matrix"                
                self.Rnn = vcv
                self.set_K_train(self.X)
                self.opt_weights(self.K_train, y)

            elif self.process == 'OU':
                # initialize Rnn with u = 0
                # uncentered
                assert np.all(np.diag(vcv) == 0.0),\
                        "OU needs path length matrix as input"

                # self.decoupleR(y, X, vcv)
                self.fullK_opt(y, X, vcv)

            else:
                raise ValueError("Model missing")

        else:
            # V is either identity or dense
            # u_curr  = self.psc
            is_PLdist = False

            if vcv is not None:
                # is path length distance matrix?
                is_PLdist = np.all(np.diag(vcv) == 0.0)
                if is_PLdist:

                    """
                    for optimizing the correlation parameter
                    we have the following objectiv function to minimize:

                    log det V + r^T V^-1 r

                    where V = A^1/2 R A^1/2 phi

                    its derivative with respect to the correlation parameter is:

                    trace(V^-1 dV/du) - r^T V^-1 dV/du V^-1 r
                    """                    
                    self.set_K_train(self.X)

                    T  = vcv.copy()
                    R = R_u(T, self.psc, center = False)
                    self.set_R_inv(R) # it returns identity matrix if vcv is None
                    self.opt_weights(self.K_train, y)

                    curr_score = self.get_new_score(None, y)
                    conv = False
                    for i in range(self.max_iter):
                        self.psc, R = self.opt_R_ls(self.psc, T)
                        self.set_R_inv(R) # it returns identity matrix if vcv is None
                        self.opt_weights(self.K_train, y)

                        new_score = self.get_new_score(None, y)
                        if (curr_score - new_score)/2 < self.tol:
                            # print(f"Converged at iteration {i + 1}, phylogenetic correlation parameter: {self.psc}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")
                            # self.psc = u_curr
                            conv = True
                            break
                        curr_score = new_score

                    if not conv:
                        print(f"Didn't phylogenetic correlation parameter: {self.psc}, new score: {new_score}, old score: {curr_score}, alpha: {np.exp(-self.psc)}")
                    # self.psc = u_curr            

            self.set_R_inv(vcv) # it returns identity matrix if vcv is None
            self.set_K_train(self.X)
            self.opt_weights(self.K_train, y)


    def check_convergence(self, f_new, f_curr, g_norm = 1, lam2 = 1, i = 1):
        """
        Check convergence of the algorithm

        f_new: updated function values
        f_curr: current function values
        tol: tolerance level

        return: True if converged, False otherwise
        """
        if i == 0:
            return False

        # self.verbose = True

        if lam2/2 <= self.tol:
            if self.verbose:
                print(f"Lambda squared is below the tolerance threshold (iter. {i})")
            return True

        if isinstance(f_new, np.ndarray) and isinstance(f_curr, np.ndarray):
            max_updt = np.linalg.norm(f_new - f_curr, ord = np.inf)
            w_max    = np.linalg.norm(f_new, ord = np.inf)
            if w_max == 0 or max_updt/w_max < self.tol:
                if self.verbose:
                    print(f"f update is below the tolerance threshold (iter. {i})")
                return True

        else:
            max_updt = np.abs(f_new - f_curr)
            w_max = np.abs(f_new)

            if max_updt < self.tol:
                if self.verbose:
                    print(f"f update is below the tolerance threshold (iter. {i})")
                return True


        if g_norm <= self.grad_tol:
            if self.verbose:
                print(f"Gradient norm is below the clipping threshold (iter. {i})")
            return True


        return False

    def f_predict(self, X_test, Rzn = None):
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert X_test is not None, "X_test should not be None"

        if self.K_train is None:
            f = self.alpha[0] * np.ones(X_test.shape[0])

        else:

            if Rzn is None:
                K_test = self.get_K(X_test, self.X, R = None, trainxtest = True)
                b, alpha = self.alpha[0], self.alpha[1:]
                return  K_test @ alpha + b

            if self.add_extraK:
                if self.process == 'OU':
                    Rzn = R_u(T = Rzn, u = self.psc, center = False)

                K_test = self.get_K(X_test, self.X, R = Rzn, trainxtest = True)
                b, alpha = self.alpha[0], self.alpha[1:]
                f =  K_test @ alpha + b # until here is ussually the way                

                if self.inv_calculated:
                    return self.conditional_latent(f, self.f_curr, self.y, Rzn)
                else:
                    return f

            else:
                if self.process == 'OU':
                    Rzn = R_u(T = Rzn, u = self.psc, center = False)

                K_test = self.get_K(X_test, self.X, R = None, trainxtest = True)
                b, alpha = self.alpha[0], self.alpha[1:]
                f =  K_test @ alpha + b
                return self.conditional_latent(f, self.f_curr, self.y, Rzn)
    
    def predict(self, X_test, R_12 = None):
        """
        Predict the response for identity link function

        Parameters
        ----------
        X_test : np.ndarray
            Test data

        Returns
        -------
        np.ndarray
            Predicted response
        """
        return self.f_predict(X_test, R_12)

    def conditional_latent(self, f_z, f_n, y, R_12):
        f_z += R_12 @ self.R_inv @ (y - f_n)
        return f_z
    
    def score(self, X_test, y_test, metric='rmse', R_12 = None):
        """
        The parameters already account for error correlation
        and the errors are assumed to be independent

        Parameters
        ----------
        X_test : np.ndarray
            test data

        y_test : np.ndarray
            test target

        metric : str, optional
            metric to use, by default 'rmse' and 'r2' for continuous
            targets and 'loss' for binary targets.

        R_12 : np.ndarray, optional
            Cross-correlation matrix between training and test data, by default None

        Returns
        -------
        float
            score
        """
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert metric in ['rmse', 'loss', 'r2', 'log_loss'], "Invalid metric"
        
        # predict function is defined 
        # in the subclass
        y_pred = self.predict(X_test, R_12)

        if metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_test) ** 2))
                    
        else:
            u = ((y_test - y_pred) ** 2).sum()
            v = ((y_test - y_test.mean()) ** 2).sum()

            return 1 - (u / v)

class Logistic(KRGEE):
    """
    Kernel Ridge Generalized Estimating Equations for 1 cluster.
    Using 1 cluster is what phylogenetic regresion deals with.

    This class is a subclass of KRGEE and implements the logistic function
    for the estimation of the mu vector and diagonal W matrix (derivative of mu).
    """
    def __init__(self,
                 kernel='rbf',
                 max_iter=300, 
                 lam=None,
                 warm_start = False,
                 tol = 0.001,
                 verbose = False,
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 add_extraK = False
                 ) -> None:
        
        super().__init__(kernel = kernel, max_iter = max_iter,
                          lam = lam, warm_start = warm_start, 
                          tol = tol, verbose = verbose, get_corr=get_corr, 
                          copy_R_inv=copy_R_inv,copy_K_train= copy_K_train, fix_phi=fix_phi,
                          add_extraK=add_extraK)

        self.reg = 'logistic'
        
        self.grad_tol = 1e-10
        self.hessian_tol = 1e-8

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):

        f_tresh = 10
        f = np.clip(f, -f_tresh, f_tresh)
        mu =  1/(1 + np.exp(-f))
        return mu
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        p1_out = self.mu(x)

        # p1 * p0, derivative of mu estimator
        self.Du = p1_out * (1 - p1_out)
        self.Du = np.clip(self.Du, self.grad_tol, 1 - self.grad_tol)
        return self.Du, p1_out
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return self.Du
    
    def conditional_latent(self, f_z, f_n, y, R_12):
        y_c = np.clip(y, self.hessian_tol, 1 - self.hessian_tol)
        # y_c = np.clip(y, 1e-3, 1 - 1e-3)
        sig_inv_y = np.log(y_c/(1 - y_c))
        e = sig_inv_y - f_n
        bias = R_12 @ self.R_inv @ e
        # print(f"e: {e[:5]}")
        # print(f"bias: {bias[:5]}")
        f_z += bias
        return f_z
    
    def predict_proba(self, X_test, R_12=None):
        f = self.f_predict(X_test, R_12)
        return self.mu(f)
    
    def predict(self, X_test, R_12=None, krigging = False):
        """
        Predict the target values for the test data
        using the fitted model.

        Parameters
        ----------
        X_test : np.ndarray
            test data

        R_12 : np.ndarray, optional
            correlation matrix, by default None

        Returns
        -------
        np.ndarray
            predicted target values
        """
        mu = self.predict_proba(X_test, R_12)
        return (mu >= 0.5).astype(float)

    def Q(self, y, u, eps=1e-12):
        """
        Get the quasi-likelihood function for the logistic model
        """
        u = np.clip(u, eps, 1 - eps)
        return (1/self.phi) * (y * np.log(u) + (1 - y) * np.log(1 - u))
    
    def score(self, X_test, y_test, R_12 = None, metric = 'log_loss', eps = 1e-3):
        """
        Score the model using the specified metric.
        Parameters
        ----------
        X_test : np.ndarray
            test data
        y_test : np.ndarray
            true labels
        R_12 : np.ndarray, optional
            correlation matrix, by default None
        metric : str, optional
            evaluation metric, by default 'log_loss'
        eps : float, optional
            small value to avoid division by zero, by default 1e-3

        Returns
        -------
        float
            evaluation score
        """

        if metric == 'loss':
            y_pred = self.predict(X_test, R_12)
            return np.mean(y_pred != y_test)
        
        elif metric == 'log_loss':
            y_pred_prob = self.predict_proba(X_test, R_12)
            # y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
            # log_losses = -( y_test*np.log(y_pred_prob) + (1 - y_test)*np.log(1 - y_pred_prob) )

            log_losses = - self.Q(y_test, y_pred_prob, eps=eps)
            return np.mean( log_losses )
    
        else:
            raise ValueError("Invalid metric. Use 'loss' or 'log_loss'.")



# class Poisson(KRGEE):
#     def __init__(self, kernel='rbf', max_iter=300, lam=0.1, warm_start=False, tol=0.001, seed=12038, verbose=False):
#         super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)

#         self.Du_mu_func = self.get_Du_mu_internal
#         self.Vu_func = self.get_Vu_func
#         self.Du = np.array([])

#     def mu(self, f):
#         return np.exp(f)
    
#     def get_Du_mu_internal(self, x):
#         """
#         Get mu vector (estimator) and D_u matrix with mu derivatives
#         from the current function values using the logistic function
#         """
#         mu = self.mu(x)
#         self.Du = mu
#         return self.Du, mu
    
#     def get_Vu_func(self, f_curr):
#         """
#         Get the variance of the estimator mu
#         """
#         return self.Du
    
#     def predict(self, X_test):
#         f =  self.f_predict(X_test)
#         return self.mu(f)
    
 
# class Gamma(KRGEE):
#     def __init__(self, kernel='rbf', max_iter=300, lam=0.1, 
#                  warm_start=False, tol=0.001, seed=12038, 
#                  verbose=False, dispersion=1):
#         super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)

#         self.Du_mu_func = self.get_Du_mu_internal
#         self.Vu_func = self.get_Vu_func
#         self.Du = np.array([])
#         self.dispersion = dispersion

#     def mu(self, f):
#         return -np.exp(-f)
    
#     def get_Du_mu_internal(self, x):
#         """
#         Get mu vector (estimator) and D_u matrix with mu derivatives
#         from the current function values using the logistic function
#         """
#         mu = self.mu(x)
#         self.Du = mu**2
#         return self.Du, mu
    
#     def get_Vu_func(self, f_curr):
#         """
#         Get the variance of the estimator mu
#         """
#         return self.Du*self.dispersion
    
#     def predict(self, X_test):
#         f =  self.f_predict(X_test)
#         return self.mu(f)
    


# region: old code


# def get_DFP(spps_train, vcv):
#     """
#     Get degrees of freedom sensu Paradis
#     Params
#     ------
#     tree: nodes, root
#     T_spps_i: list of species index in nodes
#     vcv: vcv of the species
#     """

#     n1 = spps_train[0]
#     # O(n), find the root
#     while n1.ancestor:
#         n1 = n1.ancestor
#     root = n1

#     vn = set()
#     bls = 0
#     for n in spps_train:
#         bls += n.branch_length
#         tmp_anc = n.ancestor
#         while (tmp_anc not in vn) and (tmp_anc != root):
#             vn.add(tmp_anc)
#             tmp_anc = tmp_anc.ancestor

#     for i in vn:
#         bls += i.branch_length

#     return (bls * vcv.shape[0])/np.sum(np.diag(vcv))


    # def get_V_sqInv(self, f_curr):
    #     """
    #     Get the square root of the inverse of the V matrix
    #     V = A^1/2 R A^1/2 \phi
    #     A = variance of \mu
    #     R = correlation matrix
    #     phi = dispersion parameter

    #     V^(-1/2) = ( A^1/2  R      A^1/2  \phi)^-1/2
    #              =   A^-1/4 R^-1/2 A^-1/4 \phi^-1/2 

    #     O(n^2) complexity

    #     Parameters
    #     ----------
    #     f_curr : np.ndarray
    #         current function values
    #     Returns
    #     -------
    #     np.ndarray
    #         V^(-1/2)
    #     """
    #     A_qInv = self.Vu_func(f_curr)**(-1/4)
    #     V_sqInv  = (A_qInv[:, None] * self.L_inv) * A_qInv
    #     V_sqInv *= (self.phi)**(-1/2) # O(n^2)
    #     return V_sqInv

#endregion