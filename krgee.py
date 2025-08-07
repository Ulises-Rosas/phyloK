import sys
import time
from collections import deque


import numpy as np
from phylokrr.kernels import linear_kernel, RBF_kernel

def progressbar(it, prefix="", size=60, out=sys.stdout): 
    """
    ref: https://stackoverflow.com/a/34482761
    Progress bar for loops

    """ 
    count = len(it)
    start = time.time() # time estimate start
    def show(j):
        x = int(size*j/count)
        # time estimate calculation and string
        remaining = ((time.time() - start) / j) * (count - j)        
        mins, sec = divmod(remaining, 60) # limited to minutes
        time_str = f"{int(mins):02}:{sec:03.1f}"
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
    show(0.1) # avoid div/0 
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def center_kernel(K):
    """
    Center the kernel matrix

    ref: Kernel methods in computational biology, 
    Scholkopf, Tsuda, Vert. pp. 50\\
    \\tilde K = (I - e/n)( K )(I - e/n)

    e is singular matrix of ones\\
    I is identity matrix\\
    n is the number of samples\\
    K is the kernel matrix of size n x p
    """

    n,p = K.shape
    left  = np.eye(n, dtype=np.float64) - (1/n)*np.ones((n,n), dtype=np.float64)
    right = np.eye(p, dtype=np.float64) - (1/n)*np.ones((p,p), dtype=np.float64)
    return left @ K @ right

def get_DFP(spps_train, vcv):
    """
    Get degrees of freedom sensu Paradis
    Params
    ------
    tree: nodes, root
    T_spps_i: list of species index in nodes
    vcv: vcv of the species
    """

    n1 = spps_train[0]
    # O(n), find the root
    while n1.ancestor:
        n1 = n1.ancestor
    root = n1

    vn = set()
    bls = 0
    for n in spps_train:
        bls += n.branch_length
        tmp_anc = n.ancestor
        while (tmp_anc not in vn) and (tmp_anc != root):
            vn.add(tmp_anc)
            tmp_anc = tmp_anc.ancestor

    for i in vn:
        bls += i.branch_length

    return (bls * vcv.shape[0])/np.sum(np.diag(vcv))

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
                 copy_dofp = False, # phylogenetic degrees of freedom
                 fix_phi = False
                 ) -> None:
        
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

        self.Du_mu_func = lambda x: (x, x)
        self.Vu_func = lambda x: x

        self.X = np.array([])
        self.y = np.array([])

        self.copy_R_inv = copy_R_inv        
        self.R_inv = np.array([])
        
        self.copy_K = copy_K_train
        self.K_train = np.array([])

        self.copy_dofp = copy_dofp
        self.dofp = 1

        self.phi = 1
        self.fix_phi = fix_phi
        self.r = np.array([])
        self.r_pearson = np.array([])
        self.rho = np.array([])
        self.dof = 1 # degrees of freedom

        self.f_curr = np.array([]) # current function values

        self._verbose = False
        self._update_rho = False
        self.grad_tol = 1e-14
        self.hessian_tol = 1e-10

    def __repr__(self):
        return f"KRGEE(kernel={self.kernel}, max_iter={self.max_iter}, lam={self.lam}, warm_start={self.warm_start}, tol={self.tol}, seed={self.seed})"

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

    def get_K(self, A, B):
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
            K =  RBF_kernel(A, B, self.gamma)
            return center_kernel(K)
            # return K

        elif self.kernel == 'linear':
            K =  linear_kernel(A, B, self.c)
            return center_kernel(K)
            # return K
        
        else:
            return A
        
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
        # V = V/np.linalg.norm(V, ord='fro')
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
        L = np.linalg.cholesky(R)
        L_inv = np.linalg.inv(L)
        return L_inv.T @ L_inv
        # return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(R.shape[0], dtype=R.dtype)))
        # return np.linalg.pinv(R)
    
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
        if self.copy_K and len(self.K_train) != 0:
            pass

        else:
            self.K_train = np.asfortranarray( self.get_K(X, X), dtype=X.dtype)

    def set_dofp(self, spps, vcv):
        """
        Set the phylogenetic degrees of freedom
        """
        if self.copy_dofp:
            pass

        else:
            if isinstance(spps, np.ndarray) and isinstance(vcv, np.ndarray):
                self.dofp = get_DFP(spps, vcv)

    def get_Du(self, f_curr):
        """
        Get mu vector (estimator) and W matrix with mu derivatives
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
    
    def set_df(self, K, Lpp, n):
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
        
        H = ((1/n) * K) @ Lpp # O(n^3)

        theta = np.linalg.eigvalsh(H) # O(n^3)
        theta = np.real(theta)
        df_model = np.sum(theta / (theta + self.lam))
        # phylogenetic degrees of freedom - model degrees of freedom
        # self.dof = np.clip(self.dofp -  df_model, self.grad_tol, self.dofp)
        self.dof = np.clip(n - df_model, self.grad_tol, n)

    def set_rdf(self, y, y_hat):
        """
        residual degrees of freedom
        using the covariance method

        ref: Elements of Statistical Learning, Hastie, Tibshirani, Friedman
        pp. 77, 233
        """
        # effective number of parameters
        ddof = 1
        df_model = np.cov(y_hat, y, ddof=ddof)[0, 1] / np.var(y - y_hat, ddof=ddof)
        # df_model *= self.dofp
        # self.dof = np.clip(self.dofp - df_model, self.epsilon, self.dofp)
        n = len(y_hat)
        self.dof = np.clip(n - df_model, self.grad_tol, n)

    def set_phi(self):
        
        self.phi = np.dot(self.r_pearson, self.r_pearson)/self.dof # O(n)

    def set_rho(self):

        self.rho = np.outer(self.r_pearson, self.r_pearson) # O(n^2)
        self.rho = self.rho/(self.phi * self.dof)
        # make the diagonal 1
        np.fill_diagonal(self.rho, 1)

    def krgee_fit(self, K, y):
        
        n, p = K.shape
        ones = np.ones(n, dtype=K.dtype)
        nlI = np.diag(n * self.lam * ones)

        if not self.warm_start or (np.nan in self.alpha):
            # np.random.seed(self.seed)
            # self.alpha = np.random.normal(0, np.sqrt(2/p), size=p) # alpha weights
            # self.b = np.random.normal(0, np.sqrt(2), size=1) # intercept
            self.alpha = np.zeros(p) # alpha weights
            self.b = np.zeros(1) # intercept            

            self.alpha = np.ascontiguousarray(self.alpha, dtype=K.dtype)
            self.b = np.ascontiguousarray(self.b, dtype=K.dtype)

            self.phi = 1
            self.rho = np.eye(n, dtype=K.dtype)

        assert len(self.alpha) > 0, "The model needs to be fitted first"
        
        f_curr = K @ self.alpha + self.b
        conv = False
        for i in range(self.max_iter):
            # get derivatives of estimator
            # and the inverse of the square root
            # of the variance of estimator
            Du, u = self.get_Du(f_curr)
            vu_sqrt_inv = self.get_Vu_sqrt_inv(f_curr)
            
            # D_u x V_u^(-1/2)
            Dv  = Du * vu_sqrt_inv
            # D_u x V_u^(-1/2) x R^(-1) x phi^(-1)
            DvR = Dv[:, None] * (self.R_inv * (1/self.phi))
            # error
            self.r = ( y - u ) 
            # V_u^(-1/2) x error
            self.r_pearson = vu_sqrt_inv * self.r 

            DvRve = np.dot(DvR, self.r_pearson)
            
            # alpha update
            Lp  = -DvRve + (n * self.lam * self.alpha)
            Lpp = np.matmul( DvR , (Dv[:, None] * K) ) + nlI

            # intercept update
            Lp_int  = -np.sum(DvRve)
            Lpp_int = np.sum( np.dot( DvR, (Dv * ones) ) )
            
            # newton-raphson update.
            # avoid division by zero
            if abs(Lpp_int) > self.hessian_tol and abs(Lp_int/Lpp_int) > self.hessian_tol:
                self.b -= (Lp_int/Lpp_int)

            # update alpha
            self.alpha -= np.linalg.solve(Lpp, Lp) # O(n^3)
            # self.alpha[np.abs(self.alpha) < self.grad_tol] = 0.0

            # new optimal function val
            f_new = K @ self.alpha + self.b
            if self.check_convergence(f_new, f_curr):
                if self.verbose:
                    print("Converged at update = %i" %i)
                conv = True
                break

            if self._verbose:
                print("Iteration: ", i)
                print('phi: ', self.phi)
                print('b: ', self.b)
                print('fix_phi: ', self.fix_phi)
                print('fdof: ', self.dofp)
                print()

            f_curr = f_new
        
            if not self.fix_phi:
                self.set_df(K, Lpp, n)
                # self.set_rdf(y, f_curr)
                self.set_phi()

        self.f_curr = f_curr
        self.y = y.copy()

        if self.verbose and not conv:
            print("No convergence after %i iterations" % self.max_iter)

    def fit(self, X, y, vcv = None, spps = None):
        """
        Fit the model
        """
        self.X = X # store the training data for prediction
        self.set_R_inv(vcv) # it returns identity matrix if vcv is None
        self.set_K_train(self.X)

        # it is left as 1 if vcv is None
        # this is only used for the degrees of freedom
        # and this itself is only used when fix_phi is False
        self.set_dofp(spps, vcv)

        self.krgee_fit(self.K_train, y)

    def check_convergence(self, f_new, f_curr):
        """
        Check convergence of the algorithm

        f_new: updated function values
        f_curr: current function values
        tol: tolerance level

        return: True if converged, False otherwise
        """
        max_updt = np.linalg.norm(f_new - f_curr, ord = np.inf)
        w_max    = np.linalg.norm(f_new, ord = np.inf)

        if w_max == 0 or max_updt/w_max < self.tol:
            return True
        
        return False

    def f_predict(self, X_test):
        assert len(self.alpha) > 0, "The model needs to be fitted first"

        K_test = self.get_K(X_test, self.X)
        f =  K_test @ self.alpha + self.b

        return f
    
    def score(self, X_test, y_test, metric='rmse', vcv_test = None):
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

        Returns
        -------
        float
            score
        """
        assert len(self.alpha) > 0, "The model needs to be fitted first"
        assert metric in ['rmse', 'loss', 'r2', 'log_loss'], "Invalid metric"
        
        # predict function is defined 
        # in the subclass
        y_pred = self.predict(X_test, vcv_test)

        if metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_test) ** 2))
        
        elif metric == 'loss':
            y_pred = (y_pred >= 0.5).astype(float)
            return np.mean(y_pred != y_test)
        
        elif metric == 'log_loss':
            # y_pred = np.clip(y_pred, self.hessian_tol, 1 - self.hessian_tol)
            # print(y_pred)
            eps = 1e-3
            y_pred = np.clip(y_pred, eps, 1 - eps)
            log_losses = -( y_test*np.log(y_pred) + (1 - y_test)*np.log(1 - y_pred) )
            return np.mean( log_losses )
            
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
                 lam=0.1,
                 warm_start = False,
                 tol = 0.001,
                 seed = 12038,
                 verbose = False,
                 get_corr = True,
                 copy_R_inv = False,
                 copy_K_train = False,
                 copy_dofp = False, # phylogenetic degrees of freedom
                 fix_phi = False
                 ) -> None:
        
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, 
                         verbose, get_corr, copy_R_inv, copy_K_train, copy_dofp, fix_phi)
        
        self.grad_tol = 1e-7
        self.hessian_tol = 1e-4

        self.Du_mu_func = self.get_Du_mu_internal
        self.Vu_func = self.get_Vu_func
        self.Du = np.array([])

    def mu(self, f):

        f_tresh = 30
        f = np.clip(f, -f_tresh, f_tresh)
        mu =  1/(1 + np.exp(-f))
        # mu[mu < self.grad_tol] = self.grad_tol
        # mu[mu > 1 - self.grad_tol] = 1 - self.grad_tol
        return mu
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        p1_out = self.mu(x)
        # assert np.all(p1_out > self.grad_tol) or np.all(p1_out < 1 - self.grad_tol) or np.all(p1_out != np.nan), "Diverging probabilities"
        # p1_out = np.clip(p1_out, self.grad_tol, 1 - self.grad_tol)

        # p1 * p0, derivative of mu estimator
        self.Du = p1_out * (1 - p1_out)
        self.Du = np.clip(self.Du, self.grad_tol, 1 - self.grad_tol)
        return self.Du, p1_out
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return self.Du
    
    def predict(self, X_test, R_12=None):
        f = self.f_predict(X_test)

        # mu_1 = self.mu(f)
        if R_12 is not None:
            y_c = np.clip(self.y, self.hessian_tol, 1 - self.hessian_tol)
            sig_inv_y = np.log(y_c/(1 - y_c))
            f += R_12 @ self.R_inv @ (sig_inv_y - self.f_curr)

        return self.mu(f)
    
    def score(self, X_test, y_test, vcv_test = None, metric = 'log_loss'):
        return super().score(X_test, y_test, metric=metric, vcv_test=vcv_test)


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
    
class Normal(KRGEE):
    def __init__(self, kernel='rbf', max_iter=300, lam=0.1, warm_start=False, tol=0.001, seed=12038, verbose=False):
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)

    def mu(self, f):
        return f
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        return np.ones_like(x), x
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return 1
    
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
    
class InverseGaussian(KRGEE):
    def __init__(self, kernel='rbf', max_iter=300, lam=0.1,
                  warm_start=False, tol=0.001, seed=12038, verbose=False,
                  dispersion=1):
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, verbose)
        
        self.get_Du_mu = self.get_Du_mu_internal
        self.get_Vu = self.get_Vu_func
        self.Du = np.array([])
        self.dispersion = dispersion

    def mu(self, f):
        return f**(-1/2)
    
    def get_Du_mu_internal(self, x):
        """
        Get mu vector (estimator) and D_u matrix with mu derivatives
        from the current function values using the logistic function
        """
        mu = self.mu(x)
        self.Du = (-1/2) * mu**3
        return self.Du, mu
    
    def get_Vu_func(self, f_curr):
        """
        Get the variance of the estimator mu
        """
        return (-2)*self.Du*self.dispersion

#endregion experimental work yet to be verified

