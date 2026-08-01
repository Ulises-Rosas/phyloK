import sys
import time


import numpy as np
from phylokrr.kernels import linear_kernel
# from scipy import linalg as la
from numpy import linalg as la
# from numba import njit, float64, int64

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
                 seed = 12038,
                 verbose = False,
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 add_extraK= False
                 ) -> None:

        assert max_iter >= 1, "number of iterations should be a least 1" 
        
        self.add_extraK = add_extraK
        
        self.Rnn = None
        self.Rzn = None

        self.alpha = np.array([])
        self.b = 0

        self.verbose = verbose
        self.warm_start = warm_start

        self.kernel = kernel
        self.lam = lam
        
        self.gamma = 0.1
        self.c = 0.1

        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

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
        self.rho = np.array([])
        self.dof = 1 # degrees of freedom
        self.L_inv = np.array([]) # R^-1/2


        self.f_curr = np.array([]) # current function values

        self._verbose = False
        self._update_rho = False
        self.grad_tol = 1e-10
        self.hessian_tol = 1e-10
        self.grad_clip = 2 # gradient clipping

    def __repr__(self):
        return f"KRGEE(kernel={self.kernel}, max_iter={self.max_iter}, lam={self.lam}, warm_start={self.warm_start}, tol={self.tol}, seed={self.seed})"
    
    
    def center_kernel(self, K, trainxtest = False):
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
        K_row_mean = np.mean(K, axis = 1)

        if not trainxtest:
            self.K_col_mean = np.mean(K, axis = 0)
            self.K_all_mean = np.mean(K)

        K -= self.K_col_mean
        K -= K_row_mean.reshape(-1,1)
        K += self.K_all_mean
        return K

    def set_params(self, **params):
        """
        Set parameters for the ISLEPath instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not recognized.")

    def get_params(self):
        return  {'max_iter': self.max_iter, 
                'lam': self.lam, 
                'intercept': self.b, 
                'tol': self.tol}

    def get_K(self, A, B, trainxtest=False):
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

            if self.add_extraK:
                R = self.Rzn if trainxtest else self.Rnn
            else:
                R = None

            K = RBF_kernel(A, B, self.gamma, extra_K=R)

        elif self.kernel == 'linear':
            K =  linear_kernel(A, B, self.c)

        else:
            K =  A

        return self.center_kernel(K, trainxtest)
        
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
        Get correlation matrix from vcv matrix
        """
        kr = 1/np.sqrt(np.diag(vcv))
        R = (kr[:, None] * vcv) * kr
        return R
    
    def _invR(self, R):
        """
        Inverse of the correlation matrix
        """
        # L = np.linalg.cholesky(R + np.eye(R.shape[0]) * self.hessian_tol)
        # self.L_inv = np.linalg.inv(L)

        # return self.L_inv.T @ self.L_inv

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
            self.K_train = np.asfortranarray( self.get_K(X, X), dtype=float)


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
            # np.random.seed(self.seed)
            # self.alpha = np.random.normal(size=p + 1)*np.sqrt(1./n) # alpha weights
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
                print('b: ', self.b)
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

    def krgee_fit_v2(self, K, y):
        
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
            # np.random.seed(self.seed)
            # self.alpha = np.random.normal(size=p + 1)*np.sqrt(1./n) # alpha weights
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
            e = y - u # O(n)

            # gradient
            G =  ((-1/n)*(DuDaT_V @ e)) + lK_0 @ self.alpha # O(n^2)


            try:
                d = - np.linalg.solve( H, G ) # O(n^3)

            except np.linalg.LinAlgError:
                print("Linear algebra error occurred at solve. Adding hessian_tol to the diagonal of H and trying again.")
                d =  - np.linalg.solve( H + self.hessian_tol * np.eye(H.shape[0]), G ) # O(n^3)

            g_d = G.T @ d
            lam_sq = -g_d
            
            step = self.backtracking_line_search(y, self.alpha, K_a, d, g_d)
            self.alpha += (step*d)
            
            # self.alpha += d
    
            # new optimal function val
            f_new = K_a @ self.alpha
            if self.check_convergence(f_new, f_curr, 
                                      g_norm = np.linalg.norm(G, ord=2)**2, 
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
                    # HA = ((1/n)*(DuDaT_V_DuDa[1:, 1:])) + lK # O(n^2)
                    # UHA = (DuDaT_V_DuDa[1:, 1:] + lK) # O(n^2)
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

    def backtracking_line_search(self, 
                                 y   , 
                                 w_a , 
                                 K_a , 
                                 d   , 
                                 g_d,
                                #  beta=0.01, 
                                 beta=1e-4, 
                                 p=0.5):
        """
        Also known as the Armijo's rule: It makes to satisfy 
        the first Wolfe condition. It is a sufficient decrease condition.
        """

        f_w_a = K_a @ w_a
        k_ad = K_a @ d

        b = w_a[0] # unpenalized weights 
        d_b, d_w = d[0],d[1:] # separate the unpenalized and penalized weights

        # l/2 * (w_a + t*d)^T K_0 (w_a + t*d)
        #  = l/2 * (w + t*d_w)^T K (w + t*d_w)
        #  = l/2 * (  t^2 * d_w.T K d_w + 2t w.T K d_w + w.T K w )
        #  = l/2 * (  t^2 * c1 + 2t * c2  + c3)  
        #  PEnalized term

        # c1 = d_w.T (K_a d - d_b)
        #    = d_w.T (d_b + K d_w - d_b)
        #    = d_w.T K d_w
        c1 = d_w.T @ (k_ad - d_b)

        # c2 = d_w.T (K_a w_a - b)
        #    = d_w.T (b + K w - b)  
        #    = d_w.T K w
        c2 = d_w.T @ (f_w_a - b) # d_w.T ( 1b + Kw - 1b)


        # since w.T K w is present on both sides of the inequality, 
        # it can be removed from the backtracking line search condition
        # this makes two changes:
        # 1. the PE term is now only the cross term and the quadratic term
        # 2. the obj term is now only the Q term
        
        PE = lambda x: (c1*(x**2) + 2*c2*x)*(self.lam/2)
        obj = self.Obj(y, f_w_a, 0)

        f_w_td = lambda x: f_w_a + x*k_ad

        t = 10
        # t = 1
        while self.Obj(y, f_w_td(t), PE(t)) > obj + (t * beta * g_d):
            # print(t)
            t *= p
        return t

    def fit(self, X, y, vcv = None):
        """
        Fit the model
        """
        self.X = X # store the training data for prediction
        self.set_R_inv(vcv) # it returns identity matrix if vcv is None
        self.set_K_train(self.X)

        # self.krgee_fit(self.K_train, y)
        self.krgee_fit_v2(self.K_train, y)

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

                
        max_updt = np.linalg.norm(f_new - f_curr, ord = np.inf)
        w_max    = np.linalg.norm(f_new, ord = np.inf)

        if w_max == 0 or max_updt/w_max < self.tol:
            if self.verbose:
                print(f"f update is below the tolerance threshold (iter. {i})")
            return True

        if g_norm <= self.grad_tol:
            if self.verbose:
                print(f"Gradient norm is below the clipping threshold (iter. {i})")
            return True


        return False

    def f_predict(self, X_test):
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert X_test is not None, "X_test should not be None"

        if self.K_train is None:
            f = self.alpha[0] * np.ones(X_test.shape[0])

        else:            
            K_test = self.get_K(X_test, self.X, trainxtest = True)
            b, alpha = self.alpha[0], self.alpha[1:]
            f =  K_test @ alpha + b

        return f
    
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

        f =  self.f_predict(X_test)
    
        if R_12 is not None:
            f = self.conditional_latent(f, self.f_curr, self.y, R_12)

        return f

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
                 seed = 12038,
                 verbose = False,
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 fix_phi = False,
                 add_extraK = False
                 ) -> None:
        
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, 
                         verbose, get_corr, copy_R_inv, copy_K_train, fix_phi,
                         add_extraK=add_extraK)
        
        self.grad_tol = 1e-15
        self.hessian_tol = 1e-12

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):

        f_tresh = 30
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
        f = self.f_predict(X_test)

        if R_12 is not None:
            f = self.conditional_latent(f, self.f_curr, self.y, R_12)

        return self.mu(f)
    
    def predict(self, X_test, R_12=None):
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


class Poisson(KRGEE):
    def __init__(self, kernel='rbf', max_iter=300, lam=0.1, warm_start=False, tol=0.001, seed=12038, verbose=False):
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):
        return np.exp(f)
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        mu = self.mu(x)
        self.Du = mu
        return self.Du, mu
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return self.Du
    
    def predict(self, X_test):
        f =  self.f_predict(X_test)
        return self.mu(f)
    
 
class Gamma(KRGEE):
    def __init__(self, kernel='rbf', max_iter=300, lam=0.1, 
                 warm_start=False, tol=0.001, seed=12038, 
                 verbose=False, dispersion=1):
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])
        self.dispersion = dispersion

    def mu(self, f):
        return -np.exp(-f)
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        mu = self.mu(x)
        self.Du = mu**2
        return self.Du, mu
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return self.Du*self.dispersion
    
    def predict(self, X_test):
        f =  self.f_predict(X_test)
        return self.mu(f)
    


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