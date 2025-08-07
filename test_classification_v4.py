import numpy as np
import matplotlib.pyplot as plt

from phylokrr.treeio import get_vcv,is_leaf,parseNewickTree

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


def make_circles_v5(seed, tree, test_num = 0.5, sigma=1, mu = 0, radius=0.01):
    """
    Following binaryPGLMM {ape} code for generating correlated data
    """
    np.random.seed(seed)
    # # Step 1: Define correlation structure for observations

    vcv, _ = get_vcv(tree)
    vcv = np.array(vcv)
    V = pre_process_V(vcv)
    V = get_R(V)  # Get correlation matrix
    n = V.shape[0]
    # X = np.random.normal(0, 1, (n,2) )
    X = np.random.multivariate_normal(np.zeros(n), V*sigma, 2).T
    e = np.random.multivariate_normal(np.zeros(n), V*sigma, 1).T.ravel()


    b = 0
    # eta = X[:, 0] + X[:, 1] + b  + e
    # eta = X[:, 0]*X[:, 1] + b + e
    # eta = np.sin(X[:, 0]) * np.cos(X[:, 1]) + b + e
    eta = X[:, 0]**2 + X[:, 1]**2 + b + e 
    p = 1/(1 + np.exp(-eta))  

    # random uniform distribution from 0 to 1
    # y = (np.random.uniform(0, 1, n) < p).astype(float)  # Define labels
    y = (p  <= 0.5).astype(float)  # Define labels
    

    # Step 5: Shuffle the data and split into training and testing sets
    all_index = list(range(len(vcv)))
    # np.random.shuffle(all_index)
    
    X = X[all_index,:]
    y = y[all_index]
    V = V[all_index, :][:, all_index]

    Indx = int(test_num*X.shape[0])
    test_idx = np.random.choice(all_index, Indx, replace=False)
    train_idx = list(set(all_index) - set(test_idx))
    
    test_idx = np.sort(test_idx)
    train_idx = np.sort(train_idx)
    # fine to subsample the indeces but not shuffle the data
    # logistic regression, standard one.

    X_train, X_test = X[train_idx, :], X[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    V_train = V[train_idx,:][:,train_idx]
    V_test = V[test_idx,:][:,test_idx]
    
    return X_train, X_test, y_train, y_test, V_train, V_test, train_idx, test_idx, V

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Class 0")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Class 1")
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
from cross_validation_krgee import k_fold_cv_random


tree_file = "test_tree_v3.txt"
with open(tree_file, 'r') as f: 
    tree = f.read().strip()

nodes, root = parseNewickTree(tree)
leaf_indx = {n.name: n  for n in nodes if is_leaf(n) }


vcv, spps = get_vcv(tree)
vcv = np.array(vcv)
spps = np.array(spps)


# main_seed = 12038
main_seed = 199
rng  = np.random.RandomState(seed=main_seed)

# seed = rng.randint(0, 2**20 - 1)
seed = 680589
print('seed: ', seed)
# seed = 1835838573 # for multiplication
# seed = 267837078 # for circles, 1012,227,680589,234998
cv_sample = 10
cv_folds = 5
tol = 1e-4
max_iter = 100
fix_phi = False


(X_train, X_test, 
 y_train, y_test, 
 vcv_train, vcv_test,
 train_idx, test_idx, V) = make_circles_v5(
                                 seed=seed, 
                                 tree=tree,
                                 test_num=0.5,
                                 sigma = 1,
                                 mu = 0,
                                 radius=0.6)



vcv_test_12 = V[test_idx,:][:,train_idx]
spps_train = np.array([leaf_indx[l] for l in spps[train_idx]])
spps_test  = np.array([leaf_indx[l] for l in spps[test_idx]])


print('label_ratio train:', np.sum(y_train ==1)/len(y_train))
print('label_ratio test:', np.sum(y_test ==1)/len(y_test))


print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

X_train_offset = X_train.mean(axis=0)
X_train_scale = X_train.std(axis=0)
X_train -= X_train_offset
X_train /= X_train_scale


X_test -= X_train_offset
X_test /= X_train_scale

# plot_dataset(X_train, y_train, np.array([-1, 1, -1, 1])*3)
# plt.legend()
# plt.savefig('dataset_v4.pdf', dpi=300, bbox_inches='tight')
# plot_dataset(X_test, y_test, np.array([-1, 1, -1, 1])*3)


# ref: Kernel methods in computational biology, pp. 62
params = {
    'lam' : np.logspace(-5, 15, 100, base=2),
    'gamma'  : np.logspace(-15, 3, 100, base=2),
}
# quit()

lkrr = Logistic(get_corr= False, tol=tol, 
               max_iter=max_iter, kernel='rbf', 
               verbose=True, seed=seed, 
               fix_phi = True)


import time
start = time.time()
lkrr_best_params = k_fold_cv_random(
    X_train, y_train, None, None,
    lkrr, params,
    folds=cv_folds,
    sample=cv_sample, 
    verbose=True,
    seed=seed
)
print('lkrr CV Time: ', time.time() - start)


print('lkrr best params: ', lkrr_best_params)
lkrr.set_params(**lkrr_best_params)
lkrr.fit(X_train, y_train, None, None)
lkrr_loss = lkrr.score(X_test, y_test, None,  metric='loss')
print('lkrr loss: ', lkrr_loss)
print('phi: ', lkrr.phi)
print('degrees of freedom: ', lkrr.dof)
print()

lkrgee = Logistic(get_corr= False, tol=tol, 
                 max_iter=max_iter, kernel='rbf', 
                 verbose=True, seed=seed,
                 fix_phi =  fix_phi) 


import time
start = time.time()
lkrgee_best_params = k_fold_cv_random(
    X_train, y_train, vcv_train, spps_train,
    lkrgee, params,
    folds=cv_folds,
    sample=cv_sample, 
    verbose=True, 
    seed=seed
)
print('lkrgee CV Time: ', time.time() - start)

print('lkrgee best params: ', lkrgee_best_params)
lkrgee.set_params(**lkrgee_best_params)
lkrgee._verbose =  not True
lkrgee.fit(X_train, y_train, vcv_train, None)



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]




print("log_reg intercept", log_reg.intercept_)
print("lkrr intercept: ",  lkrr.b)
print("lkrgee intercept",lkrgee.b)


from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    "C": loguniform(1e-5, 1e15),
    "gamma": loguniform(1e-15, 1e3),
}
clf_svm = RandomizedSearchCV(
    SVC(kernel="rbf", probability=True), param_grid, n_iter=10,
    n_jobs=6, cv = 5, verbose=1, random_state=12038
)

clf_svm = clf_svm.fit(X_train, y_train)

svm_pred = clf_svm.predict(X_test)
svm_deci = clf_svm.decision_function(X_test)
svm_pred_loss = np.mean(svm_pred != y_test)



from sklearn.metrics import roc_curve, auc
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
lkrr_probs    = lkrr.predict(X_test)
lkrgee_probs  = lkrgee.predict(X_test, vcv_test_12)
# print('lkrgee_probs: ', np.round(lkrgee_probs, 4)[0])
print('lkrgee_probs: ', lkrgee_probs)
# print('log_reg_probs: ', log_reg_probs)
# print('lkrgee_probs without conditioning: ', lkrgee.predict(X_test))
# print(np.round(lkrgee_probs,2))

log_reg_fpr, log_reg_tpr, _ = roc_curve(y_test, log_reg_probs, pos_label=1)
lkrr_fpr, lkrr_tpr, _ = roc_curve(y_test, lkrr_probs, pos_label=1)
lkrgee_fpr, lkrgee_tpr, _ = roc_curve(y_test, lkrgee_probs, pos_label=1)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_deci, pos_label=1)

log_reg_auc = auc(log_reg_fpr, log_reg_tpr)
lkrr_auc = auc(lkrr_fpr, lkrr_tpr)
lkrgee_auc = auc(lkrgee_fpr, lkrgee_tpr)
svm_auc = auc(svm_fpr, svm_tpr)


y_pred_log_reg = (log_reg_probs >= 0.5).astype(float)
log_reg_loss = np.mean(y_pred_log_reg != y_test)
# log_reg_loss = log_loss(y_test, log_reg_probs, eps=1e-3)
lkrr_loss = lkrr.score(X_test, y_test, None,  metric='loss')
lkrgee_loss = lkrgee.score(X_test, y_test, vcv_test_12, metric='loss')
# lkrgee_loss = log_loss(y_test, lkrgee_probs)



print()
print('log_reg_loss: ', log_reg_loss)
print('lkrr_loss: ', lkrr_loss)
print('svm loss: ', svm_pred_loss)
print('lkrgee loss: ', lkrgee_loss)
print()

print('log_reg_auc: ', log_reg_auc)
print('lkrr_auc: ', lkrr_auc)
print('lkrgee_auc: ', lkrgee_auc)
print('svm_auc: ', svm_auc)

# from sklearn.metrics import ConfusionMatrixDisplay as cf

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# plt.sca(axes[0])
# cf.from_predictions(y_test, log_reg.predict(X_test), ax=plt.gca(), cmap='Blues', 
#                   colorbar=False, normalize=None)
# plt.title('Logistic Regression Confusion Matrix')   

# plt.sca(axes[1])
# cf.from_predictions(y_test, lkrr.predict(X_test) >= 0.5, ax=plt.gca(), cmap='Blues', 
#                   colorbar=False, normalize=None)
# plt.title('Logistic KRR Confusion Matrix')  

# plt.sca(axes[2])
# cf.from_predictions(y_test, lkrgee.predict(X_test, vcv_test_12) >= 0.5, ax=plt.gca(), cmap='Blues', 
#                   colorbar=False, normalize=None)
# plt.title('Logistic KRGEE Confusion Matrix')


# # plot ROC curve

# plt.figure(figsize=(7, 5))
# plt.plot(log_reg_fpr, log_reg_tpr, label=f'Logistic Regression (AUC = {log_reg_auc:.2f})')
# # plt.plot(lkrr_fpr, lkrr_tpr, label=f'Logistic KRR (AUC = {lkrr_auc:.2f})')
# plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
# plt.plot(lkrgee_fpr, lkrgee_tpr, label=f'Logistic KRGEE (AUC = {lkrgee_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.legend()
# plt.grid(True, which='major', linestyle='--', linewidth='0.5')
# plt.tight_layout()
# plt.savefig('roc_curve_v5.pdf', dpi=300, bbox_inches='tight')




# from sklearn.metrics import brier_score_loss
# log_reg_brier = brier_score_loss(y_test, log_reg_probs)
# lkrr_brier = brier_score_loss(y_test, lkrr_probs)
# lkrgee_brier = brier_score_loss(y_test, np.round(lkrgee_probs, 3))
# print()
# print('log_reg_brier: ', log_reg_brier)
# print('lkrr_brier: ', lkrr_brier)
# print('lkrgee_brier: ', lkrgee_brier)



# from sklearn.calibration import calibration_curve
# nbins = 10
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
# # plt.hist(log_reg_probs, bins=nbins, alpha=0.5, label='Logistic Regression')
# plt.hist(lkrr_probs   , bins=nbins, alpha=0.5, label='Logistic KRR')
# plt.hist(lkrgee_probs , bins=nbins, alpha=0.5, label='Logistic KRGEE')


# # make an histogram of resdiduals
# lor_reg_res = (y_test - log_reg_probs )/np.sqrt(log_reg_probs*(1-log_reg_probs))
# lkrr_r      = (y_test - lkrr_probs    )/np.sqrt(lkrr_probs*(1-lkrr_probs))
# lkrgee_r    = (y_test - lkrgee_probs  )/np.sqrt(lkrgee_probs*(1-lkrgee_probs))

# fig, axs = plt.subplots(1, 3, figsize=(12, 6))
# # draw true network
# plt.subplot(1, 3, 1)
# plt.hist(lor_reg_res, bins=30, alpha=0.5)
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Logistic Regression Residuals")

# # plt.subplot(1, 3, 2)
# # plt.hist(lkrr_r, bins=30, alpha=0.5, label='Logistic KRR')
# # plt.xlabel("Residuals")
# # plt.ylabel("Frequency")
# # plt.title("Logistic KRR Residuals")

# plt.subplot(1, 3, 3)
# plt.hist(lkrgee_r, bins=30, alpha=0.5, label='Logistic KRGEE')
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.title("Logistic KRGEE Residuals")


# tmp_axes =  np.array([-4, 4, -4, 4])
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
# # plt.tight_layout()
# # # save figure
# # plt.savefig('../test_classification_v3.pdf', dpi=300, bbox_inches='tight')
