import sys
import time
from collections import deque

import numpy as np


def get_DFP(tree, T_spps_i, vcv):
    """
    Get degrees of freedom sensu Paradis
    Params
    ------
    tree: nodes, root
    T_spps_i: list of species index in nodes
    vcv: vcv of the species
    """
    nodes, root = tree
    vn = set()
    bls = 0
    for i in T_spps_i:
        bls += nodes[i].branch_length
        tmp_anc = nodes[i].ancestor
        while (tmp_anc not in vn) and (tmp_anc != root):
            vn.add(tmp_anc)
            tmp_anc = tmp_anc.ancestor

    for i in vn:
        bls += i.branch_length

    return (bls * vcv.shape[0])/np.sum(np.diag(vcv))

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

def k_fold_cv_vcv(X, y, vcv, model, num_folds):
    """
    k-fold cross-validation with covariance matrix
    """
    n, p = X.shape
    fold_size = n // num_folds
    mse_sum = 0

    for i in range(num_folds):

        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        vcv_train = vcv[train_idx,:][:,train_idx]
        # vcv_test = vcv[test_idx,:][:,test_idx]

        model.fit(X_train, y_train, vcv_train)
        mse_sum += model.score(X_test, y_test)

    return mse_sum / num_folds

def sample_hyperParams(params, seed, sample, verbose):
    """
    Random search for hyperparameter tuning using k-fold cross-validation

    Parameters
    ----------
    params : dict
        hyperparameters to sample from

    seed : int
        random seed

    sample : int
        number of samples

    verbose : bool
        print number of unique hyperparameters

    Returns
    -------
    np.ndarray
        sampled hyperparameters

    dict
        index of hyperparameters

    list
        names of hyperparameters
    """

    np.random.seed(seed=seed)
    # make random choice from the grid of hyperparameters
    P_n = params.keys()
    P_r = np.zeros((sample, len(P_n))) # random hyperparameters

    P_index = {}
    for n,k in enumerate(P_n):
        P_r[:,n] = np.random.choice(params[k], sample, )
        P_index[k] = n

    if verbose:
        # check tested_params are unique
        P_r = np.unique(P_r, axis=0)
        print("Number of unique hyperparameters: ", P_r.shape[0])

    return  P_r, P_index, P_n

def pick_init_index(P_index, P_r, seed):
    """
    Pick the initial index for the hyperparameters
    """

    np.random.seed(seed=seed)
    # get the lambda index
    lam_idx = ""
    if "lambda" in P_index:
        lam_idx = P_index["lambda"]

    if "lam" in P_index:
        lam_idx = P_index["lam"]

    if lam_idx:
        init_idx = np.argmax(P_r[:,lam_idx])
        # ave = np.mean(P_r[:,lam_idx])
        # init_idx = np.argmin(np.abs(P_r[:,lam_idx] - ave))
    else:
        init_idx = np.random.randint(P_r.shape[0])
        
    return init_idx

def init_model(X_train, y_train, vcv_train, model, 
                    param_init, i):
    """
    Initialize the model.
    This initialization of the model is used at the beginning of every
    fold. This set the inverse of the inverse of the correlation matrix
    and the initial alpha.
    
    if it is the first fold iteration, the alpha is calculated
    from the initial hyperparameters. Otherwise, the previous
    solution is used as the initial alpha.

    For a given fold, the correlation matrix is the same
    for all the hyperparameters, then inverse of the correlation
    matrix is calculated only once at the beginning of the fold.
    """
    # for the initial model, R inv should be calculated
    model.copy_R_inv = False
    # for the initial model, initial params should be random
    # when it is the first fold iteration. Otherwise
    # we can use the previous solution as the initial
    # i is the fold index
    model.warm_start = False if i == 0 else True

    # set the initial hyperparameters
    model.set_params(**param_init)
    # model fit, while also setting the R_inv matrix
    model.fit(X_train, y_train, vcv_train)

    # for the next iteration alpha is used
    model.warm_start = True 
    # for the next iteration R_inv is used
    model.copy_R_inv = True 

def shuffle_data(X, y, vcv, seed):
    """
    Shuffle the data
    """
    n = X.shape[0]
    all_index = list(range(n))
    np.random.seed(seed=seed)
    np.random.shuffle(all_index)
    return X[all_index, :], y[all_index], vcv[all_index,:][:,all_index]

def sort_P_r(P_r, P_index):

    if "lambda" in P_index: 
        lam_index = P_index["lambda"]

    elif "lam" in P_index:
        lam_index = P_index["lam"]

    else:
        lam_index = None

    if lam_index:
        P_r = P_r[np.argsort(P_r[:,lam_index])[::-1],]

    return P_r

def k_fold_cv_random_vcv_v2(X, y, vcv,
                     model, 
                     params,
                     folds = 3, 
                     sample = 100,
                     verbose = True,
                     seed = 12038
                     ):
    """
    Random search for hyperparameter tuning using k-fold cross-validation
    and covariance matrix
    """
    
    model.verbose = False
    np.random.seed(seed=seed)
    
    # shuffle data
    X, y, vcv = shuffle_data(X, y, vcv, seed)
    P_r, P_index, P_n = sample_hyperParams(params, seed, sample, verbose)
    
    # if lambda is present, then sort by lambda
    P_r = sort_P_r(P_r, P_index)

    # a random index if lambda is not present. 
    # otherwise, the index with the highest lambda
    init_idx = pick_init_index(P_index, P_r, seed)
    param_init = dict(zip(P_n, P_r[init_idx,:]))

    n, _ = X.shape
    fold_size = n // folds
    # linked list to store all the errors
    all_errors = deque([])
    
    # O(n^3*folds*sample)
    for i in range(folds):
        print("Fold: ", i)

        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        vcv_train = vcv[train_idx,:][:,train_idx]
        # vcv_train += np.eye(vcv_train.shape[0]) * 1e-3

        # inverse of the correlation is calculated only once
        # This is used to test all the rest of hyperparameters
        # An initial alpha is calculated from a previously 
        # specified hyperparameter. This initial alpha is
        # used for all the folds
        init_model(X_train, y_train, vcv_train,
                   model, param_init, i)
        
        init_err = model.score(X_test, y_test)

        all_errors.append([init_idx, init_err])

        for j, p_j in enumerate(P_r):
            
            if j == init_idx:
                continue

            tmp_param = dict(zip(P_n, p_j))
            
            model.set_params(**tmp_param)
            model.fit(X_train, y_train, vcv_train)
            tmp_err = model.score(X_test, y_test)
            # tmp_err = auc_score(model, X_test, y_test)
            all_errors.append([j, tmp_err])

    # O(sample*folds)
    out = {}
    for pi, err in all_errors:
        if pi in out:
            out[pi] += (err/folds)
        else:    
            out[pi] = (err/folds)
    
    # out = lkrgee_best_params # for testing
    # print("Time taken: ", time.time() - start)
    
    # O(sample * log sample)
    best_pi,best_err = sorted(out.items(), key=lambda kv: kv[1], reverse=False)[0]
    best_ = dict(zip(P_n, P_r[best_pi,:]))

    if verbose:
        print("CV score: ", best_err)

    # This let model fit with another correlation matrix
    model.copy_R_inv = False
    # let the model start with random alpha
    model.warm_start = False

    return best_

def k_fold_cv_random_vcv(X, y, vcv,
                     model, 
                     params,
                     folds = 3, 
                     sample = 500,
                     verbose = True,
                     seed = 123
                     ):
    """
    Random search for hyperparameter tuning using k-fold cross-validation
    and covariance matrix
    """

    model.verbose = False
    

    np.random.seed(seed=seed)
    # make random choice from the grid of hyperparameters
    all_params = params.keys()
    tested_params = np.zeros((sample, len(all_params)))
    for n,k in enumerate(all_params):
        tested_params[:,n] = np.random.choice(params[k], sample)

    if verbose:
        # check tested_params are unique
        tested_params = np.unique(tested_params, axis=0)
        print("Number of unique hyperparameters: ", tested_params.shape[0])
    
    all_errors = []
    for vec in progressbar(tested_params, 'CV: ', size=40):
        tmp_params = dict(zip(all_params, vec))
        model.set_params(**tmp_params)
        tmp_err = k_fold_cv_vcv(X, y, vcv, model, folds)
        all_errors.append([tmp_params, tmp_err])

    best_ = sorted(all_errors, key=lambda kv: kv[1], reverse=False)[0]

    if verbose:
        print("CV score: ", best_[1])

    return best_[0]

def k_fold_cv_random_vcv_v3(X, y, vcv,
                     tree, T_spps_train,
                     model, 
                     params,
                     folds = 3, 
                     sample = 100,
                     verbose = True,
                     seed = 12038
                     ):
    """
    Random search for hyperparameter tuning using k-fold cross-validation
    and covariance matrix
    """
    
    model.verbose = False
    np.random.seed(seed=seed)
    
    # shuffle data
    X, y, vcv = shuffle_data(X, y, vcv, seed)
    P_r, P_index, P_n = sample_hyperParams(params, seed, sample, verbose)
    
    # if lambda is present, then sort by lambda
    P_r = sort_P_r(P_r, P_index)

    # a random index if lambda is not present. 
    # otherwise, the index with the highest lambda
    init_idx = pick_init_index(P_index, P_r, seed)
    param_init = dict(zip(P_n, P_r[init_idx,:]))

    n, _ = X.shape
    fold_size = n // folds
    # linked list to store all the errors
    all_errors = deque([])
    
    # O(n^3*folds*sample)
    for i in range(folds):
        print("Fold: ", i)

        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        vcv_train = vcv[train_idx,:][:,train_idx]
        dof = get_DFP(tree, T_spps_train[train_idx], vcv_train)
        model.dofp = dof
        # print('degrees of freedom from phylo: ', dof)
    
        # inverse of the correlation is calculated only once
        # This is used to test all the rest of hyperparameters
        # An initial alpha is calculated from a previously 
        # specified hyperparameter. This initial alpha is
        # used for all the folds
        init_model(X_train, y_train, vcv_train,
                   model, param_init, i)
        
        init_err = model.score(X_test, y_test)
        all_errors.append([init_idx, init_err])

        for j, p_j in enumerate(P_r):
            
            if j == init_idx:
                continue

            tmp_param = dict(zip(P_n, p_j))
            
            model.set_params(**tmp_param)
            model.fit(X_train, y_train, vcv_train)
            tmp_err = model.score(X_test, y_test)
            all_errors.append([j, tmp_err])

    # O(sample*folds)
    out = {}
    for pi, err in all_errors:
        if pi in out:
            out[pi] += (err/folds)
        else:    
            out[pi] = (err/folds)
    
    # out = lkrgee_best_params # for testing
    # print("Time taken: ", time.time() - start)
    
    # O(sample * log sample)
    best_pi,best_err = sorted(out.items(), key=lambda kv: kv[1], reverse=False)[0]
    best_ = dict(zip(P_n, P_r[best_pi,:]))

    if verbose:
        print("CV score: ", best_err)

    # This let model fit with another correlation matrix
    model.copy_R_inv = False
    # let the model start with random alpha
    model.warm_start = False

    return best_
