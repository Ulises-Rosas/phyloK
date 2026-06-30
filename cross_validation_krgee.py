import sys
import time
from collections import deque

import numpy as np
from alpha_beta_weighting import weighted_sample

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
    # model.warm_start = False if i == 0 else True
    model.warm_start = False

    # set the initial hyperparameters
    model.set_params(**param_init)
    # model fit, while also setting the R_inv matrix
    time_start = time.time()
    model.fit(X_train, y_train, vcv_train)
    time_end = time.time()
    print(f"Initial model fit time: {time_end - time_start:.2f} seconds")

    # for the next iteration alpha is used
    model.warm_start = False

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

    shuffle_vcv = vcv[all_index,:][:,all_index] if isinstance(vcv, np.ndarray) else None
    return X[all_index, :], y[all_index], shuffle_vcv

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

def random_folds(n, k, fold_size, sort_folds = True, rng = None):
    """
    sequencial random division of the data into k folds.

    It randomly divides the data into k folds, where each fold
    contains an sorted list of indices.

    O(k n log n) algorithm
    """
    all_index = np.array(list(range(n)))
    rng.shuffle(all_index)
    # even though we could also use random choice
    # it does not guarantee the indeces are going to
    # be sorted.

    # random division of the data
    myfolds = deque([])
    # O(folds*n log n) if sort_folds is True
    # O(folds*n) if sort_folds is False
    for i in range(k):
        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))
        # get the indeces, which are unordered due to shuffling
        test_idx  = all_index[test_idx] 
        train_idx = all_index[train_idx]

        if sort_folds:
            # and then sort them
            # O(n log n)
            test_idx  = np.sort(all_index[test_idx] )
            train_idx = np.sort(all_index[train_idx])

        myfolds.append([train_idx, test_idx])
    return myfolds

def k_fold_cv_random(X, y, vcv, model, 
                   params,
                   folds = 3, 
                   sample = 100,
                   verbose = True,
                   seed = 12038):
    """
    Random search for hyperparameter tuning using k-fold cross-validation
    and covariance matrix

    leave one-out exist in the realm of phylogenetic
    regressions.

    set taxa randomly. Pick clades too.
    """
    
    model.verbose = False
    np.random.seed(seed=seed)
    rng = np.random.RandomState(seed=seed)
    
    # X = X_train
    # X_train, y, vcv = shuffle_data(X, y, vcv, seed)
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
    
    # O(n log n), generates sorted indices for
    # each fold
    myfolds = random_folds(n, folds, fold_size, 
                           sort_folds = True, rng = rng)
    
    # O(n^3*folds*sample)
    for i, (train_idx, test_idx) in enumerate(myfolds):
        print("Fold: ", i)
        # print(test_idx)

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        vcv_train  = vcv[train_idx,:][:,train_idx] if isinstance(vcv, np.ndarray) else None
        vcv_test_12 = vcv[test_idx,:][:,train_idx] if isinstance(vcv, np.ndarray) else None
        
        # inverse of the correlation is calculated only once
        # This is used to test all the rest of hyperparameters
        # An initial alpha is calculated from a previously 
        # specified hyperparameter. This initial alpha is
        # used for all the folds
        init_model(X_train, y_train, vcv_train,
                   model, param_init, i)
        
        init_err = model.score(X_test, y_test, vcv_test_12)*len(y_test)
        all_errors.append([init_idx, init_err])

        for j, p_j in enumerate(P_r):
            
            if j == init_idx:
                continue

            tmp_param = dict(zip(P_n, p_j))
            
            model.set_params(**tmp_param)
            # vcv and spps are already copied
            # from init_model
            model.fit(X_train, y_train)
            tmp_err = model.score(X_test, y_test, vcv_test_12)*len(y_test)
            all_errors.append([j, tmp_err])

    # O(sample*folds)
    out = {}
    for pi, err in all_errors:
        if pi in out:
            out[pi] += (err/n)
        else:    
            out[pi] = (err/n)
    
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


def k_fold_cv_weigthed(X, y, vcv, model, W,
                       params,
                       folds = 3, 
                       sample = 100,
                       verbose = True,
                       seed = 12038):
    """
    """
    
    model.verbose = False
    np.random.seed(seed=seed)
    rng = np.random.RandomState(seed=seed)
    
    # X = X_train
    P_r, P_index, P_n = sample_hyperParams(params, seed, sample, verbose)
    
    # if lambda is present, then sort by lambda
    P_r = sort_P_r(P_r, P_index)

    # a random index if lambda is not present. 
    # otherwise (most typical), the index with the highest lambda
    init_idx = pick_init_index(P_index, P_r, seed)
    param_init = dict(zip(P_n, P_r[init_idx,:]))

    n, _ = X.shape
    # linked list to store all the errors
    all_errors = deque([])
    myfolds = weighted_sample(W, rng, folds=folds)

    # O(n^3*folds*sample)
    for i, (train_idx, test_idx) in enumerate(myfolds):
        print("Fold: ", i)
        # print(test_idx)
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        is_vcv =  isinstance(vcv, np.ndarray)
        vcv_train = None
        vcv_test_12 = None

        if is_vcv and model.add_extraK:
            model.Rnn = vcv[train_idx,:][:,train_idx]
            model.Rzn = vcv[test_idx,:][:,train_idx]

        elif is_vcv and not model.add_extraK:
            vcv_train = vcv[train_idx,:][:,train_idx]
            vcv_test_12 = vcv[test_idx,:][:,train_idx]

        elif not is_vcv and model.add_extraK:
            raise ValueError("Missing correlation structure at Cross validation")
        
        else:
            # everything is None
            pass

        # inverse of the correlation is calculated only once
        # This is used to test all the rest of hyperparameters
        # An initial alpha is calculated from a previously 
        # specified hyperparameter. This initial alpha is
        # used for all the folds
        init_model(X_train, y_train, vcv_train,
                   model, param_init, i)
        
        # init_err1 = model.score(X_test, y_test, vcv_test_12)*len(y_test)
        # init_err2 = model.score(X_test, y_test, None)*len(y_test)
        # init_err = max(init_err1, init_err2)

        init_err = model.score(X_test, y_test, None)*len(y_test)

        all_errors.append([init_idx, init_err])

        j = 0
        for p_j in progressbar(P_r, prefix = "Testing hyperparameters: ", size=60):
            
            if j == init_idx:
                continue

            tmp_param = dict(zip(P_n, p_j))
            
            model.set_params(**tmp_param)
            # vcv and spps are already copied
            # from init_model
            model.fit(X_train, y_train, vcv_train)

            # tmp_err1 = model.score(X_test, y_test, vcv_test_12)*len(y_test)
            # tmp_err2 = model.score(X_test, y_test, None)*len(y_test)
            # tmp_err = max(tmp_err1, tmp_err2)

            tmp_err = model.score(X_test, y_test, None)*len(y_test)

            all_errors.append([j, tmp_err])
            j += 1

    # O(sample*folds)
    out = {}
    for pi, err in all_errors:
        if pi in out:
            out[pi] += (err/n)
        else:    
            out[pi] = (err/n)
    
    # out = lkrgee_best_params # for testing
    # print("Time taken: ", time.time() - start)
    
    # O(sample * log sample)
    best_pi,best_err = sorted(out.items(), key=lambda kv: kv[1], reverse=False)[0]
    best_ = dict(zip(P_n, P_r[best_pi,:]))

    if verbose:
        print("CV score: ", best_err)

    # This let model fit with another correlation matrix
    model.copy_R_inv = False
    # let the model start with new alpha
    model.warm_start = False
    
    return best_
