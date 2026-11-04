import numpy as np

def check_convergence(x_old, x_new, threshold=1e-6):
    max_updt = np.linalg.norm(x_new - x_old, ord = np.inf)
    w_max    = np.linalg.norm(x_new, ord = np.inf)
    if w_max == 0.0 or max_updt/w_max <= threshold:
        return True
    
    return False

def get_Options(X, nan_value = 0):
    if np.isnan(nan_value):
        Oi,Oj = np.where(~np.isnan(X))
    else:
        Oi,Oj = np.where(X != nan_value)

    return np.c_[Oi, Oj]

def init_AB(k, shape, rng = np.random.default_rng(12038)):
    a,b = shape

    A = rng.normal(size = (a,k))
    B = rng.normal(size = (k,b))
    return A, B

def ALS_legacy(sample_train, sample_test, 
                k = 6, lam1 = 0.1, lam2 = 0.1, 
                get_errors = True, epsilon = 1e-6,
                max_epochs = 1000, 
                rng = np.random.default_rng(12038)):

    Oij_test = get_Options(sample_test)
    Oij = get_Options(sample_train)

    M_test = sample_test[Oij_test[:,0], Oij_test[:,1]]

    Ik  = np.eye(k)
    A,B = init_AB(k, sample_train.shape, rng = rng)
    
    all_errors = []
    for _ in range(max_epochs):
        Ac = np.copy(A)
        Bc = np.copy(B)
        # row and columns
        # update
        for i,j in Oij:
            # i,j
            Ok_ = Oij[ Oij[:,0] == i ]
            Mkj = sample_train[ Ok_[:,0], Ok_[:,1] ]
            Bj = B[:, Ok_[:,1] ]

            # matrix form of
            # update rule of A[i:]
            A[i,:] =  np.linalg.solve(Bj @ Bj.T + lam1*Ik, Bj @ Mkj.T).T

            O_k = Oij[ Oij[:,1] == j ]
            Mik = sample_train[ O_k[:,0], O_k[:,1] ]    
            Ai = A[ O_k[:,0],:]

            # matrix form of
            # update rule of B[:k]
            B[:,j] = np.linalg.solve( Ai.T @ Ai + lam2*Ik, Ai.T @ Mik )

        if get_errors:
            # matrix form of
            # error evaluation
            AiBj = (A @ B)[Oij_test[:,0], Oij_test[:,1]]
            # rmse's
            myrmse = np.sqrt( np.mean( (M_test - AiBj)**2 ) )
            # print("current RMSE: ", myrmse)
            all_errors.append( myrmse)

        if check_convergence(Ac, A, epsilon) and\
           check_convergence(Bc, B, epsilon):
            break
    
    AiBj = (A @ B)[ Oij_test[:,0], Oij_test[:,1] ]
    final_error = np.sqrt( np.mean( (M_test - AiBj)**2 ) )
    
    return A, B, all_errors, final_error


def ALS(M_train, 
        k = 6, lam1 = 0.1, lam2 = 0.1, 
        epsilon = 1e-6,
        max_epochs = 1000, 
        rng = np.random.default_rng(12038),
        nan_value = 0):


    Oij = get_Options(M_train, nan_value = nan_value)
    Ik  = np.eye(k)
    A,B = init_AB(k, M_train.shape, rng = rng)
    
    for _ in range(max_epochs):
        Ac = np.copy(A)
        Bc = np.copy(B)
        # row and columns
        # update
        for i,j in Oij:
            # i,j
            Ok_ = Oij[ Oij[:,0] == i ]
            Mkj = M_train[ Ok_[:,0], Ok_[:,1] ]
            Bj = B[:, Ok_[:,1] ]

            # matrix form of
            # update rule of A[i:]
            A[i,:] =  np.linalg.solve(Bj @ Bj.T + lam1*Ik, Bj @ Mkj.T).T

            O_k = Oij[ Oij[:,1] == j ]
            Mik = M_train[ O_k[:,0], O_k[:,1] ]    
            Ai = A[ O_k[:,0],:]

            # matrix form of
            # update rule of B[:k]
            B[:,j] = np.linalg.solve( Ai.T @ Ai + lam2*Ik, Ai.T @ Mik )

        if check_convergence(Ac, A, epsilon) and\
           check_convergence(Bc, B, epsilon):
            break
    
    return A, B

def score(A, B, M_test, nan_value = 0):
    Oij_test = get_Options(M_test, nan_value = nan_value)
    M_test = M_test[Oij_test[:,0], Oij_test[:,1]]
    AiBj = (A @ B)[Oij_test[:,0], Oij_test[:,1]]
    final_error = np.sqrt( np.mean( (M_test - AiBj)**2 ) )
    return final_error


if __name__ == "__main__":

    train_file = "../data/rate_train.csv"
    test_file  = "../data/rate_test.csv"

    rng = np.random.default_rng(12038)

    sample_train = np.loadtxt(train_file, delimiter=',')
    sample_test  = np.loadtxt(test_file , delimiter=',')

    lam1 = 4
    lam2 = 4
    k = 5

    (A,B) = ALS(sample_train,
                k = k, lam1 = lam1, lam2 = lam2, 
                max_epochs = 100, epsilon = 1e-3, 
                rng = rng)

    final_error = score(A, B, sample_test)
    print("Final RMSE: ", final_error) # Final RMSE:  0.8042507213383732


    
    # (A,B,_,_) = ALS_legacy(sample_train, sample_test, 
    #                     k = k, lam1 = lam1, lam2 = lam2, 
    #                     max_epochs = 100, epsilon = 1e-3, 
    #                     get_errors = False,rng = rng,)


    # import matplotlib.pyplot as plt
    # plt.plot(all_errors)
    # plt.xlabel('Epoch')
    # plt.ylabel('RMSE')
    # plt.title('Convergence of ALS')
    # plt.show()


    # with open("./completed_matrix.csv", "w") as f:
    #     np.savetxt(f, A @ B, delimiter=",", fmt='%1.4f')
