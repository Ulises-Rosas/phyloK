import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky



from phylokrr.treeio import get_vcv
from phylokrr.treeio import parseNewickTree


def pre_process_V(V):
    n, _ = V.shape
    V = V/np.max(V)
    detV_n = np.linalg.det(V)**(1/n)
    print('detV_n: ', detV_n)
    V = V/detV_n if detV_n != 0 else V
    return V


def bin_error(n, V):
    eta = np.random.multivariate_normal(n, V, 1).T.ravel()
    return 1/(1 + np.exp(-eta))

def make_circles_v3(seed, tree, test_num = 0.5, sigma=1, mu = 0, radius=0.01):

    np.random.seed(seed)

    vcv,_ = get_vcv(tree)
    vcv = np.array(vcv)

    # kr = 1/np.sqrt(np.diag(vcv))
    # V = (kr[:, None] * vcv) * kr
    V = pre_process_V(vcv)
    # # Step 2: Generate independent Gaussian features
    # # noise = 0.1  # Add noise to the observations
    num_features = 2
    # num_samples = V.shape[0]
    # X_uncorrelated = np.random.randn(num_samples, num_features)

    # Step 3: correlate observations
    # X = (L @ X_uncorrelated)   # Now observations are correlated
    X = np.random.multivariate_normal(np.zeros(vcv.shape[0]) + mu, V*sigma,  num_features).T
    corr_err = np.random.multivariate_normal(np.zeros(vcv.shape[0]), V, 1).T.ravel()

    # Step 4: Define a **circular non-linearity** (donut shape)
    # radius sets the radius for the decision boundary

    decision_boundary = np.sqrt(X[:, 0]**2 + X[:, 1]**2)  # Compute distance from origin
    y = (decision_boundary  + corr_err  < radius).astype(int)  # Define labels
    # y = (decision_boundary < radius).astype(int)  # Define labels

    # Step 5: Shuffle the data and split into training and testing sets
    all_index = list(range(len(vcv)))
    np.random.shuffle(all_index)
    
    X = X[all_index,:]
    y = y[all_index]
    V = V[all_index, :][:, all_index]

    Indx = int(test_num*X.shape[0])
    test_idx = np.random.choice(all_index, Indx, replace=False)
    train_idx = list(set(all_index) - set(test_idx))

    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    V_train = V[train_idx,:][:,train_idx]
    V_test = V[test_idx,:][:,test_idx]
    
    return X_train, X_test, y_train, y_test, V_train, V_test, train_idx, test_idx

def make_circles_v4(seed, tree, test_num = 0.5, sigma=1, mu = 0, radius=0.01):
    np.random.seed(seed)
    # # Step 1: Define correlation structure for observations

    vcv, _ = get_vcv(tree)
    vcv = np.array(vcv)

    # kr = 1/np.sqrt(np.diag(vcv))
    # vcv = (kr[:, None] * vcv) * kr
    # num_samples = V.shape[0]

    # V = pre_process_V(vcv)
    V = vcv.copy()
    n = V.shape[0]

    X_iid = np.random.normal(0, 1, (n,2) )

    E,Q = np.linalg.eig( V )
    L = Q @ np.diag( E**(1/2) )

    X =  L @ X_iid  # Now observations are correlated
    X = np.real(X)  # Ensure no complex numbers
    
    b = 0
    f_xi = X[:,0]*X[:,1] + b

    # f_xi = L @ f_xi
    # f_xi = np.real(f_xi)  # Ensure no complex numbers

    p = 1/(1 + np.exp(-f_xi))  


    y = (p >= 0.5).astype(float)  # Define labels
    # corr_err = np.random.multivariate_normal(np.zeros(n), V*sigma, 1).T.ravel()
    # y = (p + corr_err >= 0.5).astype(float)  # Define labels


    # Step 5: Shuffle the data and split into training and testing sets
    all_index = list(range(len(vcv)))
    np.random.shuffle(all_index)
    
    X = X[all_index,:]
    y = y[all_index]
    V = V[all_index, :][:, all_index]

    Indx = int(test_num*X.shape[0])
    test_idx = np.random.choice(all_index, Indx, replace=False)
    train_idx = list(set(all_index) - set(test_idx))

    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    V_train = V[train_idx,:][:,train_idx]
    V_test = V[test_idx,:][:,test_idx]
    
    return X_train, X_test, y_train, y_test, V_train, V_test, train_idx, test_idx


def make_circles_v5(seed, tree, test_num = 0.5, sigma=1, mu = 0, radius=0.01):
    """
    Following binaryPGLMM {ape} code for generating correlated data
    """
    np.random.seed(seed)
    # # Step 1: Define correlation structure for observations

    vcv, _ = get_vcv(tree)
    vcv = np.array(vcv)
    V = pre_process_V(vcv)
    V *= sigma
    # V = vcv.copy()
    n = V.shape[0]
    # X = np.random.normal(0, 1, (n,2) )
    X = np.random.multivariate_normal(np.zeros(n), V, 2).T

    # E, Q = np.linalg.eig(V)
    # L = Q @ np.diag(E**(1/2))
    # L = np.linalg.cholesky(V)
    # correlated err
    # L.shape
    # e = L @ np.random.normal(0, 1, n )
    e = np.random.multivariate_normal(np.zeros(n), V, 1).T.ravel()


    b = 0.5
    # eta = X[:, 0]**2 + X[:, 1]**2 + b  + e
    eta = X[:, 0]*X[:, 1] + b  + e
    p = 1/(1 + np.exp(-eta))  

    # random uniform distribution from 0 to 1
    # y = (np.random.uniform(0, 1, n) < p).astype(float)  # Define labels
    y = (p >= 0.5).astype(float)  # Define labels
    

    # Step 5: Shuffle the data and split into training and testing sets
    all_index = list(range(len(vcv)))
    np.random.shuffle(all_index)
    
    X = X[all_index,:]
    y = y[all_index]
    V = V[all_index, :][:, all_index]

    Indx = int(test_num*X.shape[0])
    test_idx = np.random.choice(all_index, Indx, replace=False)
    train_idx = list(set(all_index) - set(test_idx))

    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    V_train = V[train_idx,:][:,train_idx]
    V_test = V[test_idx,:][:,test_idx]
    
    return X_train, X_test, y_train, y_test, V_train, V_test, train_idx, test_idx

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='major', linestyle='--', linewidth='0.5')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.bwr , alpha=0.3)

from krgee import Logistic
import matplotlib.pyplot as plt
from cross_validation_krgee import (k_fold_cv_random_vcv_v2, 
                                    k_fold_cv_random_vcv_v3)




tree_file = "test_tree_v3.txt"
with open(tree_file, 'r') as f: 
    tree = f.read().strip()

nodes, root = parseNewickTree(tree)
vcv, spps = get_vcv(tree)
vcv = np.array(vcv)

T_spps_i = np.array([i.index for i in nodes if i.name])


seed = 12034
cv_sample = 50
cv_folds = 3
tol = 1e-3
max_iter = 200


(X_train, X_test, 
 y_train, y_test, 
 vcv_train, vcv_test,
 train_idx, test_idx) = make_circles_v5(
                                 seed=seed, 
                                 tree=tree,
                                 test_num=0.5,
                                 sigma = 1,
                                 mu = 0,
                                # radius=1.25
                                radius=1.
                                 )

T_spps_train = T_spps_i[train_idx]
T_spps_test = T_spps_i[test_idx]


print('label_ratio train:', np.sum(y_train ==1)/len(y_train))
print('label_ratio test:', np.sum(y_test ==1)/len(y_test))


print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

X_train_offset = X_train.mean(axis=0)
X_train_scale = X_train.std(axis=0)
X_train -= X_train_offset
X_train /= X_train_scale


# X_test -= X_train_offset
# X_test /= X_train_scale

X_test = X_test - np.mean(X_test, axis=0)
X_test = X_test / np.std(X_test, axis=0)


plot_dataset(X_train, y_train, np.array([-1, 1, -1, 1])*3)
# plot_dataset(X_test, y_test, np.array([-1, 1, -1, 1])*3)



params = {
    'lambda' : np.logspace(-5, 3, 200, base=2),
    'gamma'  : np.logspace(-5, 3, 200, base=2),
}
# quit()

I_train = np.eye(X_train.shape[0])
lkrr = Logistic(get_corr=False, tol=tol, 
               max_iter=max_iter, kernel='rbf', 
               verbose=True, seed=seed, 
               fix_phi =  True)
# lkrr.phi = 1 


import time
start = time.time()
lkrr_best_params = k_fold_cv_random_vcv_v2(
    X_train, y_train, I_train, 
    lkrr, params,
    folds=cv_folds,
    sample=cv_sample, 
    verbose=True, 
    seed=seed
)
print('lkrr CV Time: ', time.time() - start)


print('lkrr best params: ', lkrr_best_params)
lkrr.set_params(**lkrr_best_params)
lkrr.fit(X_train, y_train, I_train)
lkrr_loss = lkrr.score(X_test, y_test)
print('lkrr loss: ', lkrr_loss)
print('phi: ', lkrr.phi)
print('degrees of freedom: ', lkrr.dof)
print()


lkrgee = Logistic(get_corr=False, tol=tol, 
                 max_iter=max_iter, kernel='rbf', 
                 verbose=True, seed=seed, 
                 fix_phi = False)

import time
start = time.time()
lkrgee_best_params = k_fold_cv_random_vcv_v3(
    X_train, y_train, vcv_train, 
    (nodes, root), T_spps_train,
    lkrgee, params,
    folds=cv_folds,
    sample=cv_sample, 
    verbose=True, 
    seed=seed
)
print('lkrgee CV Time: ', time.time() - start)

print('lkrgee best params: ', lkrgee_best_params)
lkrgee.set_params(**lkrgee_best_params)
lkrgee.verbose = True
lkrgee.fit(X_train, y_train, vcv_train)
lkrgee_loss = lkrgee.score(X_test, y_test)
print('lkrgee_loss testing: ', lkrgee_loss)
print('phi: ', lkrgee.phi)
print('degrees of freedom: ', lkrgee.dof)



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# log loss
from sklearn.metrics import log_loss
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
log_reg_loss = log_loss(y_test, log_reg_probs)
# log_reg_loss = 1 -  log_reg.score(X_test, y_test)



# lkrr.b
# lkrgee.b
# log_reg.intercept_

# tmp_axes =  [-4, 4, -4, 4]
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# plt.sca(axes[0])
# plt.title(f'Logistic Regression (log loss: {round(log_reg_loss, 3)})')
# plot_predictions(log_reg, tmp_axes)
# plot_dataset(X_test, y_test, tmp_axes)

# plt.sca(axes[1])
# plt.title(f'Logistic KRR (log loss: {round(lkrr_loss, 3)})')
# plot_predictions(lkrr, tmp_axes)
# plot_dataset(X_test, y_test, tmp_axes)

# plt.sca(axes[2])
# plt.title(f'Logistic KRGEE (log loss: {round(lkrgee_loss, 3)})')
# plot_predictions(lkrgee, tmp_axes)
# plot_dataset(X_test, y_test, tmp_axes)
# plt.tight_layout()



# # # # ROC curve
from sklearn.metrics import roc_curve, auc
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
lkrr_probs    = lkrr.predict(X_test)
lkrgee_probs  = lkrgee.predict(X_test)

log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg_probs, pos_label=1)
lkrr_fpr, lkrr_tpr, _ = roc_curve(y_test, lkrr_probs, pos_label=1)
lkrgee_fpr, lkrgee_tpr, _ = roc_curve(y_test, lkrgee_probs, pos_label=1)

log_reg_auc = auc(log_reg_fpr, log_reg_tpr)
lkrr_auc = auc(lkrr_fpr, lkrr_tpr)
lkrgee_auc = auc(lkrgee_fpr, lkrgee_tpr)




print()
print('log_reg_loss: ', log_reg_loss)
print('lkrr_loss: ', lkrr_loss)
print('lkrgee loss: ', lkrgee_loss)
print()

print('log_reg_auc: ', log_reg_auc)
print('lkrr_auc: ', lkrr_auc)
print('lkrgee_auc: ', lkrgee_auc)


from sklearn.metrics import brier_score_loss
log_reg_brier = brier_score_loss(y_test, log_reg_probs)
lkrr_brier = brier_score_loss(y_test, lkrr_probs)
lkrgee_brier = brier_score_loss(y_test, lkrgee_probs)
print()
print('log_reg_brier: ', log_reg_brier)
print('lkrr_brier: ', lkrr_brier)
print('lkrgee_brier: ', lkrgee_brier)
# # plot ROC curve

# from sklearn.calibration import calibration_curve
# nbins = 5
# prob_true_log_reg, prob_pred_log_reg= calibration_curve(y_test, log_reg_probs, n_bins=nbins)
# prob_true_lkrr   , prob_pred_lkrr   = calibration_curve(y_test, lkrr_probs   , n_bins=nbins)
# prob_true_lkrgee , prob_pred_lkrgee = calibration_curve(y_test, lkrgee_probs , n_bins=nbins)
# plt.figure(figsize=(7, 5))
# plt.plot(prob_pred_log_reg, prob_true_log_reg, "s-", label="Logistic Regression")
# plt.plot(prob_pred_lkrr, prob_true_lkrr, "o-", label="Logistic KRR")
# plt.plot(prob_pred_lkrgee, prob_true_lkrgee, "^-", label="Logistic KRGEE")
# plt.plot([0, 1], [0, 1], "k--")
# plt.xlabel("Predicted Probability")
# plt.ylabel("True Probability")
# plt.title("Calibration Curve")
# plt.legend()


# nbins = 3
# plt.hist(log_reg_probs, bins=nbins, alpha=0.5, label='Logistic Regression')
# plt.hist(lkrr_probs   , bins=nbins, alpha=0.5, label='Logistic KRR')
# plt.hist(lkrgee_probs , bins=nbins, alpha=0.5, label='Logistic KRGEE')


# # make an histogram of resdiduals
lor_reg_res = (y_test - log_reg_probs )/np.sqrt(log_reg_probs*(1-log_reg_probs))
lkrr_r      = (y_test - lkrr_probs    )/np.sqrt(lkrr_probs*(1-lkrr_probs))
lkrgee_r    = (y_test - lkrgee_probs  )/np.sqrt(lkrgee_probs*(1-lkrgee_probs))

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
# draw true network
plt.subplot(1, 3, 1)
plt.hist(lor_reg_res, bins=30, alpha=0.5)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Logistic Regression Residuals")

plt.subplot(1, 3, 2)
plt.hist(lkrr_r, bins=30, alpha=0.5, label='Logistic KRR')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Logistic KRR Residuals")

plt.subplot(1, 3, 3)
plt.hist(lkrgee_r, bins=30, alpha=0.5, label='Logistic KRGEE')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Logistic KRGEE Residuals")
