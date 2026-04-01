import sys
import time


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

        # self.copy_dofp = copy_dofp

        self.phi = 1
        self.fix_phi = fix_phi
        self.r = np.array([])
        self.rho = np.array([])
        self.dof = 1 # degrees of freedom
        self.L_inv = np.array([]) # R^-1/2


        self.f_curr = np.array([]) # current function values

        self._verbose = False
        self._update_rho = False
        self.grad_tol = 1e-14
        self.hessian_tol = 1e-10
        self.grad_clip = 2 # gradient clipping

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
        self.L_inv = np.linalg.inv(L)
        return self.L_inv.T @ self.L_inv
    
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
        if self.copy_K and len(self.K_train) != 0:
            pass

        else:
            self.K_train = np.asfortranarray( self.get_K(X, X), dtype=X.dtype)


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

    def get_V_sqInv(self, f_curr):
        """
        Get the square root of the inverse of the V matrix
        V = A^1/2 R A^1/2 \phi
        A = variance of \mu
        R = correlation matrix
        phi = dispersion parameter

        V^(-1/2) = ( A^1/2  R      A^1/2  \phi)^-1/2
                 =   A^-1/4 R^-1/2 A^-1/4 \phi^-1/2 

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
        A_qInv = self.Vu_func(f_curr)**(-1/4)
        V_sqInv  = (A_qInv[:, None] * self.L_inv) * A_qInv
        V_sqInv *= (self.phi)**(-1/2) # O(n^2)
        return V_sqInv
    
    def get_V_Inv(self, f_curr):
        """
        Get the square root of the inverse of the V matrix
        V = A^1/2 R A^1/2 \phi
        A = variance of \mu
        R = correlation matrix
        phi = dispersion parameter

        V^(-1/2) = ( A^1/2  R      A^1/2  \phi)^-1
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
        
        n, p = K.shape
        ones = np.ones(n, dtype=K.dtype)

        K_0 = np.zeros((n + 1, p + 1), dtype=K.dtype)
        K_0[1:, 1:] = K
        lK_0 = self.lam * K_0
        lK = self.lam * K
        # concatenate ones to K for intercept
        K_a = np.column_stack((ones, K))
        K_a = np.asfortranarray(K_a, dtype=K.dtype)

        if not self.warm_start or (np.nan in self.alpha):
            # np.random.seed(self.seed)
            # self.alpha = np.random.normal(size=p + 1)*np.sqrt(1./n) # alpha weights

            self.alpha = np.zeros(p + 1) # alpha weights
            self.alpha = np.ascontiguousarray(self.alpha, dtype=K.dtype)
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
            # UH =  (DuDaT_V_DuDa + lK_0) # O(n^2)
            e = y - u # O(n)

            # gradient
            G =  ((-1/n)*(DuDaT_V @ e)) + lK_0 @ self.alpha # O(n^2)

            # gradient clipping, O(n)
            # to avoid exploding gradients
            G_norm = np.linalg.norm(G, ord=2)
            if G_norm >= self.grad_clip:
                G *= (self.grad_clip / G_norm)

            self.alpha -= np.linalg.solve( H, G ) # O(n^3)

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
                #  hessian of the penalized coefficients (alpha without intercept)
                HA = ((1/n)*(DuDaT_V_DuDa[1:, 1:])) + lK # O(n^2)
                # UHA = (DuDaT_V_DuDa[1:, 1:] + lK) # O(n^2)

                self.set_df(HA) # O(n^3)
                self.set_phi(e)

            f_curr = f_new

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
        b, alpha = self.alpha[0], self.alpha[1:]
        f =  K_test @ alpha + b

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
                    
        else:
            u = ((y_test - y_pred) ** 2).sum()
            v = ((y_test - y_test.mean()) ** 2).sum()

            return 1 - (u / v)

# self = KRGEE()

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
                 fix_phi = False
                 ) -> None:
        
        super().__init__(kernel, max_iter, lam, warm_start, tol, seed, 
                         verbose, get_corr, copy_R_inv, copy_K_train, fix_phi)
        
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
    
    def predict_proba(self, X_test, R_12=None):
        f = self.f_predict(X_test)

        # mu_1 = self.mu(f)
        if R_12 is not None:
            y_c = np.clip(self.y, self.hessian_tol, 1 - self.hessian_tol)
            sig_inv_y = np.log(y_c/(1 - y_c))

            f += R_12 @ self.R_inv @ (sig_inv_y - self.f_curr)

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
    
    def score(self, X_test, y_test, vcv_test = None, metric = 'log_loss', eps = 1e-3):
        """
        Score the model using the specified metric.
        Parameters
        ----------
        X_test : np.ndarray
            test data
        y_test : np.ndarray
            true labels
        vcv_test : np.ndarray, optional
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
            y_pred = self.predict(X_test, vcv_test)
            return np.mean(y_pred != y_test)
        
        elif metric == 'log_loss':
            y_pred_prob = self.predict_proba(X_test, vcv_test)
            y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)

            log_losses = -( y_test*np.log(y_pred_prob) + (1 - y_test)*np.log(1 - y_pred_prob) )
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

