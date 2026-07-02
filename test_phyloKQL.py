import os
import numpy as np
import matplotlib.pyplot as plt

from alpha_beta_weighting import (alpha_weighting,
                                  split_data_idx)

from block_cross_validation import k_fold_cv_weigthed

from phylokrr.treeio import (get_vcv, 
                             parseNewickTree, 
                             is_leaf, get_adj, get_C, 
                             get_APSP)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def run_rscript(X_train, y_train, train_idx, vcv):

    rscript_file_path = f"/Users/ulises/Desktop/Machine\ Learning/HW3/SantaquiteriasProblem"
    X_train_f = f"X_train_{seed}.csv"
    y_train_f = f"y_train_{seed}.csv"
    index_train_f = f"index_train_{seed}.csv"
    vcv_train_f = f"VCV_train{seed}.csv"
    out_f = f"output_{seed}.csv"

    np.savetxt(    X_train_f,   X_train, delimiter=",")
    np.savetxt(    y_train_f,   y_train, delimiter=",")
    np.savetxt(index_train_f,   train_idx, delimiter=",")
    np.savetxt(  vcv_train_f, vcv[train_idx,:][:,train_idx], delimiter=",")


    os.system("ulimit -s 16384 # enlarge stack limit to 16 megs")
    script = os.path.join(rscript_file_path, "logReg.R")
    os.system(f"Rscript {script} {vcv_train_f}\
                    {index_train_f} {X_train_f}\
                    {y_train_f} {out_f}")

    out = []
    with open(out_f, 'r') as f:
        for i in f.readlines():
            r_output = i.strip()
            if r_output == 'beta':
                continue
            out.append(float(r_output))

    os.system(f"rm {X_train_f} {y_train_f} {index_train_f} {vcv_train_f} {out_f}")
    return out


def pre_process_V(V):
    n, _ = V.shape
    V = V/np.max(V)
    detV_n = np.linalg.det(V)**(1/n)
    print('detV_n: ', detV_n)
    V = V/detV_n if detV_n != 0 else V
    return V

def get_R( vcv):
    """
    Get correlation matrix from vcv matrix
    """
    kr = 1/np.sqrt(np.diag(vcv))
    R = (kr[:, None] * vcv) * kr
    return R


def get_D(mytree,):
    nodes, root = parseNewickTree(mytree)
    assert not isinstance(root.right, list), "Unrooted tree not supported"
    leaves = [n for n in nodes if is_leaf(n)]
    # names = [n.name for n in leaves]

    adj = get_adj(nodes) # O(n)
    C = get_C(nodes, adj) # O(n)

    # O(n^2), it transverses the tree n times, and
    # each traversal is O(n)
    D = get_APSP(leaves, adj, C)
    # print(D)
    return D

def get_from_file(leaves, filename = "./traits_R.csv"):
    
    n = len(leaves)
    from_R = {}
    with open(filename, 'r') as f:
        for i in f.readlines():
            a,b,c = i.split(',')
            if  a == '""':
                continue
            from_R[a.replace('"','')] = [float(b),float(c)]

    k = 0
    X = np.zeros((n, 2))
    for l in leaves:
        X[k,:] = from_R[l]
        k += 1

    return X

def simTraitsR(tree_file, leaves, sigma2, alpha, seed = 12038):
    script = '/Users/ulises/Desktop/SLL/KRGEE/rTrait_script.R'
    os.system(f'Rscript {script} {tree_file} {sigma2} {alpha} {seed}')

    X = get_from_file(leaves, "./traits_R.csv")
    os.system('rm traits_R.csv')
    return X

def get_vcvR():
    
    V_file = "./V.csv"
    names = []
    V = []
    k = 0
    with open(V_file, 'r') as f:
        for i in f.readlines():
            a = i.strip().split(',')
            if k == 0:
                names.append(a[1:])      
            else:
                x_i = np.array([float(j) for j in a[1:]])
                V.append(x_i)
            k+=1

    V = np.asarray(V)
    return V


def make_circles_v5(seed, tree, test_num = 0.5, 
                    sigma=1, alpha_ou = 0.5,
                    split_weight = 1/100,
                    tree_file = 'test_tree_v5.txt',
                    sigma_e = 1, alpha_ou_e = 0.5):
    """
    Following binaryPGLMM {ape} code for generating correlated data
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    
    # # Step 1: Define correlation structure for observations
    # get_vcv is a deterministic function, so leaves1 and leaves2 should be the same
    V, leaves1 = get_vcv(tree, process="OU", sigma2=sigma, alpha=alpha_ou)
    Ve, leaves2 = get_vcv(tree, process="OU", sigma2=sigma_e, alpha=alpha_ou_e)

    V = np.round(V, decimals=4)  # Round to 3 decimals to avoid numerical issues
    n = V.shape[0]
    # V = np.eye(n)
    # print("V:", V[0:5, 0:5])

    # off diagonal max elements of V
    # off_diag_values = V - np.diag(np.diag(V))    
    # print("Off-diagonal max of V:", np.max(off_diag_values))
    # print("Off-diagonal min of V:", np.min(off_diag_values))
    # print("Off-diagonal median of V:", np.median(off_diag_values))
    # print("Off-diagonal mean of V:", np.mean(off_diag_values))

    print()
    # V = vcv
    # V = pre_process_V(V)
    # print("Pre-processed V: ", V[1:5, 1:5])
    

    R = get_R(V)  # Get correlation matrix
    # R = V  # Get correlation matrix
    # print("R: ", R[0:5, 0:5])

    # off diagonal max elements of R
    # R_off_diag_vals = R - np.diag(np.diag(R))
    # print("Off-diagonal max of R:", np.max(R_off_diag_vals))
    # print("Off-diagonal min of R:", np.min(R_off_diag_vals))
    # print("Off-diagonal median of R:", np.median(R_off_diag_vals))
    # print("Off-diagonal mean of R:", np.mean(R_off_diag_vals))


    X = np.random.multivariate_normal(np.zeros(n), V, 2).T
    # X = simTraitsR(tree_file, leaves1, sigma2=sigma, alpha=alpha_ou, seed=seed)
    # print("X: ", X[0:5, 0:5])

    e = np.random.multivariate_normal(np.zeros(n), Ve, 1).T.ravel()
    # e = simTraitsR(tree_file, leaves1, sigma2=sigma_e, alpha=alpha_ou_e, seed=seed)[:,1]
    # print("e: ", e[0:5])

    # b = -2.8
    # b = -1.5
    # b = -0.0
    # eta = X[:, 0] + X[:, 1] + b  + e

    # sine + cosine pattern
    # b = -1.5
    # eta = 1.5*np.sin(4*X[:, 0]) + 1.5*np.cos(4*X[:, 1]) + b + e

    # Spiral pattern
    b = -0.0
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = np.arctan2(X[:, 1], X[:, 0])
    eta = np.sin(3 * theta - r) + b + e

    # Sine wave pattern
    # b = 0.0    
    # eta = X[:,1] - 2*np.sin(3.5*X[:, 0]) + b + e

    # checkerboard pattern
    # b = -0.0
    # eta = np.cos(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + b + e

    # XOR pattern
    # b = -0.5
    # eta = X[:, 0]*X[:, 1] + b + e
    # eta_r = eta.reshape(-1, 1)
    # print("eta_r empirical: ", eta_r[0:5, 0:5])
    # V_empirical = eta_r @ eta_r.T 


    # b = 0.5
    # # eta = 5 * np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 1] + b + e
    # eta = 5 * np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + b + e

    # circle 1    
    # b = -2.5
    # # b = -2.1
    # b = -1
    # b = -.04
    # eta = X[:, 0]**2 + X[:, 1]**2 + b + e

    # circle 2
    # b = 0.1
    # eta = 1.25 - np.sqrt(X[:, 0]**2 + X[:, 1]**2) + b + e

    p = 1/(1 + np.exp(-eta))
    # random uniform distribution from 0 to 1
    # y  = rng.binomial(1, p).astype(float)
    y  = (p >= 0.5).astype(float)

    Indx = int(test_num*X.shape[0])
    
    A = alpha_weighting(tree, alpha=split_weight)
    # match order of rows using leaves1
    P = np.array([A[leaf] for leaf in leaves1])

    test_idx, train_idx = split_data_idx(Indx, P, rng)

    return X, y, train_idx, test_idx, R, P

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Class 0")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Class 1")
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14)
    plt.axis(axes)
    plt.grid(True, which='major', linestyle='--', linewidth='0.5')

def conditional_latent(model, f_z, f_n, y, R_12):
    y_c = np.clip(y, model.hessian_tol, 1 - model.hessian_tol)
    # y_c = np.clip(y, 1e-2, 1 - 1e-2)
    sig_inv_y = np.log(y_c/(1 - y_c))
    f_z += R_12 @ model.R_inv @ (sig_inv_y - f_n)
    return f_z

def phylocorrected_logistic(lkrgee, Bphyolm, X_test, X_train, y_train, vcv_test_12):
    eta_z = Bphyolm[0] + Bphyolm[1]*X_test[:,0] + Bphyolm[2]*X_test[:,1]
    
    eta_n = Bphyolm[0] + Bphyolm[1]*X_train[:,0] + Bphyolm[2]*X_train[:,1]
    eta_zn = conditional_latent(lkrgee, eta_z, eta_n, y_train, vcv_test_12)
    return np.exp(eta_zn)/(1+np.exp(eta_zn))
    # return np.exp(eta_z)/(1+np.exp(eta_z))


# # ref: Kernel methods in computational biology, pp. 62
params = {
    'lam' : np.logspace(-5, 15, 100, base=2),
    'gamma'  : np.logspace(-15, 3, 100, base=2),
}


from krgee_v2 import Logistic
import matplotlib.pyplot as plt

v = 6
tree_file = f"test_tree_v{v}.txt"
with open(tree_file, 'r') as f: 
    tree = f.read().strip()

D = get_D(tree)
t_max = np.max(D)

plot_results = True


seed = 12038
print('seed: ', seed)
fix_phi = True

cv_sample = 100 # number of hyperparameters to test
cv_folds = 5
tol = 1e-3
max_iter = 100
test_num = 0.3

M1 = .5 # The lower, the more correlation
M2 = .5 # The lower, the more correlation

c1 = 1 # variance constant.
c2 = 0.09

split_weight = 2 # novelty |log(split_weight)|

# C1 = [.5]
# C2 = np.linspace(0.1, 2, num = 3)

C1 = np.linspace(0.1, 2, num = 3)
C2 = np.linspace(0.1, 2, num = 3)


lkrr = Logistic(get_corr= False, tol=tol, 
                max_iter=max_iter, kernel='rbf', 
                verbose=True, seed=seed, 
                fix_phi = True, add_extraK=True)

lkrgee = Logistic(get_corr = False, tol = tol, 
                max_iter = max_iter, kernel = 'rbf', 
                verbose = True, seed = seed,
                fix_phi = fix_phi)

log_reg = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)

out = {}

for c1 in C1:
    for c2 in C2:

        alpha_ou_e = M2 / t_max
        sigma_e = 2 * alpha_ou_e * c2

        alpha_ou = M1 / t_max
        sigma = 2 * alpha_ou * c1


        # param_str = f'auc_curve_c{c1}_{c2}_tmax{M1:.3f}_{M2:.3f}_splitw{split_weight:.3f}_testNum{test_num:.3f}_noise{sigma_e:.1f}_Circl1_v{v}_seed{seed}'

        print("alpha_ou: ", alpha_ou)
        print("sigma: ", sigma)

        (X, y, train_idx, 
        test_idx, V, W) = make_circles_v5(
                            seed=seed,
                            tree=tree,
                            tree_file=tree_file,
                            test_num=test_num,
                            sigma =sigma,
                            alpha_ou = alpha_ou,
                            split_weight=split_weight,
                            sigma_e = sigma_e,
                            alpha_ou_e = alpha_ou_e)


        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        vcv_train = V[train_idx,:][:,train_idx]
        vcv_test_12 = V[test_idx,:][:,train_idx]

        test_ratio =  np.sum(y_test == 1)/len(y_test)
        train_ratio = np.sum(y_train == 1)/len(y_train)



        print()
        print('label_ratio train:', train_ratio)
        print('label_ratio test:', test_ratio)

        print('X_train shape: ', X_train.shape)
        print('X_test shape: ', X_test.shape)

        if not ( 0.2 < test_ratio < 0.8):
            continue


        X_train_offset = X_train.mean(axis=0)
        X_train_scale = X_train.std(axis=0)

        X_train -= X_train_offset
        X_train /= X_train_scale

        X_test -= X_train_offset
        X_test /= X_train_scale

        # vcv, spps = get_vcv(tree, process="OU")
        # Bphyolm = run_rscript(X_train, y_train, train_idx, vcv)

        import time
        start = time.time()
        lkrr_best_params = k_fold_cv_weigthed(
            X_train, y_train, vcv_train,
            lkrr,  W[train_idx], params,
            folds=cv_folds,
            sample=cv_sample, 
            verbose=True, 
            seed=seed
        )
        print('lkrr (+ phylo K) CV Time: ', time.time() - start)
        print('lkrr best params: ', lkrr_best_params)


        import time
        start = time.time()
        lkrgee_best_params = k_fold_cv_weigthed(
            X_train, y_train, vcv_train,
            lkrgee,  W[train_idx], params,
            folds=cv_folds,
            sample=cv_sample, 
            verbose=True, 
            seed=seed
        )
        print('lkrgee CV Time: ', time.time() - start)
        print('lkrgee best params: ', lkrgee_best_params)

        lkrgee.set_params(**lkrgee_best_params)
        lkrgee.fit(X_train, y_train, vcv_train)

        lkrgee_loss = lkrgee.score(X_test, y_test, vcv_test_12)
        print('lkrgee loss: ', lkrgee_loss)

        lkrr.set_params(**lkrr_best_params)
        lkrr.Rnn = vcv_train
        lkrr.Rzn = vcv_test_12


        lkrr.fit(X_train, y_train, None)
        lkrr_loss = lkrr.score(X_test, y_test, None)
        print('lkrr (+phylo K) loss: ', lkrr_loss)

        log_reg.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)


        log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
        lkrgee_probs = lkrgee.predict_proba(X_test, None)
        lkrgee_probs_corrected  = lkrgee.predict_proba(X_test, vcv_test_12)
        lkrr_probs = lkrr.predict_proba(X_test, None)
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        # Plogreg_probs = phylocorrected_logistic(lkrgee, Bphyolm, 
        #                                         X_test, X_train,
        #                                         y_train, 
        #                                         vcv_test_12)


        log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg_probs, pos_label=1)
        lkrgee_fpr, lkrgee_tpr, _ = roc_curve(y_test, lkrgee_probs, pos_label=1)
        lkrgee_fpr_corrected, lkrgee_tpr_corrected, _ = roc_curve(y_test, lkrgee_probs_corrected, pos_label=1)
        lkrr_fpr, lkrr_tpr, _ = roc_curve(y_test, lkrr_probs, pos_label=1)
        rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs, pos_label=1)
        # Plogreg_fpr, Plogreg_tpr, _ = roc_curve(y_test, Plogreg_probs, pos_label=1)



        log_reg_auc = auc(log_reg_fpr, log_reg_tpr)
        lkrgee_auc = auc(lkrgee_fpr, lkrgee_tpr)
        lkrgee_auc_corrected = auc(lkrgee_fpr_corrected, lkrgee_tpr_corrected)
        lkrr_auc = auc(lkrr_fpr, lkrr_tpr)
        rf_auc = auc(rf_fpr, rf_tpr)
        # Plogreg_auc = auc(Plogreg_fpr, Plogreg_tpr)

        # out[(c1,c2)] = [log_reg_auc, Plogreg_auc, rf_auc, lkrr_auc, lkrgee_auc, lkrgee_auc_corrected]
        out[(c1,c2)] = [log_reg_auc, None, rf_auc, 
                        lkrr_auc, lkrgee_auc, lkrgee_auc_corrected]

        # print("log_reg_auc: ",log_reg_auc)
        # print("Plogreg_auc: ",Plogreg_auc)
        print("rf_auc: ",rf_auc)
        print()
        print("lkrr_auc (engineered kernel): ",lkrr_auc)
        print()
        print("lkrgee_auc: ",lkrgee_auc)
        print("lkrgee_auc (adjusted): ",lkrgee_auc_corrected)

# print(out)
import json
with open(f"auc_curve_c{c1}_{c2}_tmax{M1:.3f}_{M2:.3f}_splitw{split_weight:.3f}_testNum{test_num:.3f}_noise{sigma_e:.1f}_Circl1_v{v}_seed{seed}.json", 'w') as f:
    json.dump(out, f, indent=4)


# out = {(np.float64(0.1), np.float64(0.1)): [np.float64(0.5311709224752703), None, np.float64(0.5304807913503565), np.float64(0.5489993098688751), np.float64(0.4454796411318151), np.float64(0.4683689901081206)], (np.float64(0.1), np.float64(0.575)): [np.float64(0.5573062810274121), None, np.float64(0.5085797539391324), np.float64(0.5383121087848046), np.float64(0.5436002590114397), np.float64(0.4917979710770559)], (np.float64(0.1), np.float64(1.05)): [np.float64(0.5899532710280374), None, np.float64(0.5079651656754461), np.float64(0.576677994902294), np.float64(0.42491503823279525), np.float64(0.5404630416312659)], (np.float64(0.1), np.float64(1.525)): [np.float64(0.5807716769132923), None, np.float64(0.5271358914564342), np.float64(0.5794996820012719), np.float64(0.4258002967988128), np.float64(0.531375874496502)], (np.float64(0.1), np.float64(2.0)): [np.float64(0.6025993237531698), None, np.float64(0.5669907016060862), np.float64(0.5760777683854608), np.float64(0.6015426880811496), np.float64(0.5290574809805579)], (np.float64(0.325), np.float64(0.1)): [np.float64(0.6542183883862824), None, np.float64(0.5502840311382285), np.float64(0.6201346517988638), np.float64(0.4538186408584052), np.float64(0.5061014096360195)], (np.float64(0.325), np.float64(0.575)): [np.float64(0.6184015263938945), None, np.float64(0.554748781004876), np.float64(0.5625397498410005), np.float64(0.5965656137375451), np.float64(0.5259698961204154)], (np.float64(0.325), np.float64(1.05)): [np.float64(0.603809523809524), None, np.float64(0.551058201058201), np.float64(0.5892063492063492), np.float64(0.6004232804232804), np.float64(0.5688888888888889)], (np.float64(0.325), np.float64(1.525)): [np.float64(0.6070071760236387), None, np.float64(0.5935521317011397), np.float64(0.5917053609117771), np.float64(0.5948712536935415), np.float64(0.5791473195441115)], (np.float64(0.325), np.float64(2.0)): [np.float64(0.6045012679628063), None, np.float64(0.5907650042265428), np.float64(0.569737954353339), np.float64(0.6206677937447168), np.float64(0.5981614539306848)], (np.float64(0.55), np.float64(0.1)): [np.float64(0.733110295704727), None, np.float64(0.6701381394344916), np.float64(0.7471400820202893), np.float64(0.7149794949276925), np.float64(0.6078135117634362)], (np.float64(0.55), np.float64(0.575)): [np.float64(0.7453430258973195), None, np.float64(0.698886869604725), np.float64(0.725465697410268), np.float64(0.7392094502498865), np.float64(0.6176737846433439)], (np.float64(0.55), np.float64(1.05)): [np.float64(0.737610004631774), None, np.float64(0.6849235757295045), np.float64(0.7308939323761), np.float64(0.7307781380268643), np.float64(0.6223946271421955)], (np.float64(0.55), np.float64(1.525)): [np.float64(0.718742857142857), None, np.float64(0.6921714285714284), np.float64(0.7148571428571429), np.float64(0.6990857142857143), np.float64(0.6453714285714286)], (np.float64(0.55), np.float64(2.0)): [np.float64(0.718044263775971), None, np.float64(0.6809507678410118), np.float64(0.716237579042457), np.float64(0.693992773261066), np.float64(0.6097560975609756)], (np.float64(0.775), np.float64(0.1)): [np.float64(0.7796515523788251), None, np.float64(0.7556957784230512), np.float64(0.7790931427295062), np.float64(0.7750725932544115), np.float64(0.6622738440920259)], (np.float64(0.775), np.float64(0.575)): [np.float64(0.7493540051679587), None, np.float64(0.7504698144233029), np.float64(0.7486492835330044), np.float64(0.7513507164669956), np.float64(0.6447028423772609)], (np.float64(0.775), np.float64(1.05)): [np.float64(0.7250887573964497), None, np.float64(0.7076331360946746), np.float64(0.7513609467455621), np.float64(0.6688757396449705), np.float64(0.658224852071006)], (np.float64(0.775), np.float64(1.525)): [np.float64(0.7114964831433421), None, np.float64(0.6933664807179238), np.float64(0.753941304875091), np.float64(0.6306087800145525), np.float64(0.669476109628911)], (np.float64(0.775), np.float64(2.0)): [np.float64(0.7210780574624969), None, np.float64(0.690376303076532), np.float64(0.7345537757437071), np.float64(0.720188151538266), np.float64(0.6313246885329266)], (np.float64(1.0), np.float64(0.1)): [np.float64(0.777285447761194), None, np.float64(0.7579874067164178), np.float64(0.7976912313432836), np.float64(0.7739039179104478), np.float64(0.7686567164179104)], (np.float64(1.0), np.float64(0.575)): [np.float64(0.7643884892086331), None, np.float64(0.7372816032887975), np.float64(0.7825025693730729), np.float64(0.7652877697841727), np.float64(0.7231500513874615)], (np.float64(1.0), np.float64(1.05)): [np.float64(0.771210927239296), None, np.float64(0.7391646966115051), np.float64(0.7959022852639874), np.float64(0.7742316784869976), np.float64(0.7423167848699763)], (np.float64(1.0), np.float64(1.525)): [np.float64(0.7620249800690938), None, np.float64(0.6947249534945522), np.float64(0.7674727610948711), np.float64(0.7658782885995216), np.float64(0.6931969173531757)], (np.float64(1.0), np.float64(2.0)): [np.float64(0.7420541108484371), None, np.float64(0.6634489099028106), np.float64(0.7491463094299974), np.float64(0.7433674809561335), np.float64(0.672314158129761)]}

# print("out: ", out)

# OUT = {(np.float64(0.1), np.float64(0.1)): [np.float64(0.9636019356196086), None, np.float64(0.9592362718283189), np.float64(0.971807279612876), np.float64(0.9462444771723122), np.float64(0.8905954134231012)], (np.float64(0.1), np.float64(1.05)): [np.float64(0.7923913043478261), None, np.float64(0.7790760869565218), np.float64(0.8116304347826088), np.float64(0.7730434782608696), np.float64(0.9047826086956522)], (np.float64(0.1), np.float64(2.0)): [np.float64(0.7174008327854482), None, np.float64(0.6726386149463073), np.float64(0.7353714661406969), np.float64(0.6757615603769449), np.float64(0.9005588428665351)], (np.float64(1.05), np.float64(1.05)): [np.float64(0.6807013625195444), None, np.float64(0.7297297297297298), np.float64(0.8003127094036184), np.float64(0.5212195666741121), np.float64(0.7826669644851463)], (np.float64(1.05), np.float64(2.0)): [np.float64(0.7239804587935429), None, np.float64(0.7011469838572644), np.float64(0.7950297366185217), np.float64(0.5676508071367885), np.float64(0.8070305862361937)], (np.float64(2.0), np.float64(0.1)): [np.float64(0.7968302883557121), None, np.float64(0.985031917235307), np.float64(0.9869029275808936), np.float64(0.7266123706801672), np.float64(0.889060092449923)], (np.float64(2.0), np.float64(1.05)): [np.float64(0.7706315789473684), None, np.float64(0.891), np.float64(0.9325263157894736), np.float64(0.7598947368421052), np.float64(0.9115263157894737)], (np.float64(2.0), np.float64(2.0)): [np.float64(0.7217878275570583), None, np.float64(0.8158812341504649), np.float64(0.8727810650887573), np.float64(0.26901944209636514), np.float64(0.8953930684699914)]}

# for i in OUT.keys():
#     print(i)

# x_unique = sorted(set([i[0] for i in OUT.keys()]))
# y_unique = sorted(set([i[1] for i in OUT.keys()]))

# X,Y = np.meshgrid(x_unique, y_unique)

# Z = np.zeros(X.shape)
# for i in range(len(y_unique)):
#     for j in range(len(x_unique)):
#         x_val = x_unique[j]
#         y_val = y_unique[i]

#         if (x_val, y_val) in OUT:
#             # 0: log_reg_auc, 1: None, 2: rf_auc, 
#             # 3: lkrr_auc, 4: lkrgee_auc, 
#             # 5: lkrgee_auc_corrected
#             # auc = OUT[(x_val, y_val)][3]
#             auc = OUT[(x_val, y_val)][5] - OUT[(x_val, y_val)][2]
#         else:
#             auc = None

#         Z[i,j] = auc

# plt.figure(figsize=(6,5))
# contour = plt.contourf(X,Y,Z, levels = 6, cmp = 'viridis')
# plt.colorbar(contour, label = 'AUC')
# plt.xlabel(r'$\sigma^2_{x}$')
# plt.ylabel(r'$\sigma^2_{\epsilon}$')
# # plt.savefig(f'./tmp_imgs/contour_diffAUC_lkrr_{split_weight}_noR_splitBugFree_noBernoulli.pdf', dpi=300, bbox_inches='tight')
