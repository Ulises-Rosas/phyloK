"""
phyloK on a small simulated example: the circle function on a 500-taxon tree.

Reproduces the 'final' model of the paper (defaults of phyloKQL_fit_v2.py):
OU tree-biased kernel, correction during CV ('shrink'), residual scaling
('std'), and AGOP with a diagonal metric (5 iterations, greedy). As with
--tree_file in phyloKQL_fit_v2.py, the training tree is given to the model,
so the correction solves R_nn^{-1} e on the tree (O(n)) during the CV and
the final fit instead of with a dense O(n^3) solve.

The data (circle_data.npz) were simulated with simulate_circle.py and are
loaded rather than simulated, so the output is the same on every machine.

Run from this folder:
    python circle_final.py
"""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from phylokrr.treeio import parseNewickTree, is_leaf, get_adj, get_C, get_APSP
from phyloK.block_cross_validation import k_fold_cv_weigthed
from phyloK.phyloK_v2 import Logistic
from phyloK.tree_solve import get_training_tree

if __name__ == "__main__":

    seed = 12038

    # ------------------------------------------------------------------ 1. tree
    with open("test_tree_v5.txt") as f:
        newick = f.readline().strip()

    nodes, root = parseNewickTree(newick)
    leaves = [n for n in nodes if is_leaf(n)]
    names = [n.name for n in leaves]                 # row order of X, y and T
    adj = get_adj(nodes)
    C = get_C(nodes, adj)
    D = get_APSP(leaves, adj, C)                      # path lengths between tips, O(n^2)
    n = len(leaves)

    # ------------------------------------- 2. circle data and phylogeny-informed split
    # simulated by simulate_circle.py (predictors and errors under Brownian motion,
    # y = 1 if Logistic(1.35 - sqrt(x1^2 + 1.5 x2^2) + e) >= 1/2, split with rho = 5)
    data = np.load("circle_data.npz")
    assert list(data["names"]) == names, "circle_data.npz does not match the tree"
    X, y = data["X"], data["y"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    T = 2 * D / np.max(D)                             # rescaled path lengths: r_ij = exp(-alpha t_ij)
    T_train = T[np.ix_(train_idx, train_idx)]
    T_test_train = T[np.ix_(test_idx, train_idx)]

    # training tree: branch lengths on the scale of T (divided by the tree height,
    # max(D) / 2), pruned to the training taxa, in the row order of X_train
    tree_nodes, _ = parseNewickTree(newick)
    height = np.max(D) / 2
    for node in tree_nodes:
        node.branch_length /= height
    spps_train = np.array(names)[train_idx]
    tree_train = get_training_tree(tree_nodes, spps_train)   # (nodes, root)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    offset, scale = X_train.mean(0), X_train.std(0)   # standardize with the training taxa
    X_train = (X_train - offset) / scale
    X_test = (X_test - offset) / scale
    print(f"{n} taxa: {len(train_idx)} training, {len(test_idx)} test "
        f"(class 1: {y_train.mean():.2f} train, {y_test.mean():.2f} test)")

    # --------------------------------------------------------- 4. the 'final' model
    model = Logistic(tol=1e-6, max_iter=100, kernel='rbf', verbose=False,
                    fix_phi=True, get_corr=False)
    model.process = 'OU'               # tree-biased kernel k = k_x + exp(-alpha t)
    model.use_quasi_newton = True      # L-BFGS-B
    model.do_corr = True               # correction of Eq. 6 during CV
    model.corr_type = 'shrink'
    model.corr_scale_error = 'std'     # residuals scaled by their standard deviation
    model.agop_iter = 5
    model.agop_nogreedy = False        # keep a new M only if it lowers the objective
    model.memory_qn = len(train_idx)

    # hyperparameters: 100 log-uniform samples, five folds
    params = {
        'lam':   np.logspace(-12, 3, 100, base=2),
        'gamma': np.logspace(-15, 4, 100, base=2),
        'corr':  np.logspace(-10, 0, 100, base=2),
    }
    W_train = np.full(len(train_idx), 1 / len(train_idx))   # uniform fold sampling
    start = time.time()
    best = k_fold_cv_weigthed(X_train, y_train, T_train, model, W_train, params,
                            folds=5, sample=100, verbose=False, seed=seed, cores=5,
                            nodes=tree_train[0], leaves=spps_train)  # tree pruned per fold
    print(f"CV ({time.time() - start:.0f} s), best hyperparameters:",
        {k: float(f"{v:.3g}") for k, v in best.items()})

    model.set_params(**best)
    model.set_training_tree(tree_train, spps_train)   # tree-based solve in the fit and the prediction
    model.do_agop = True               # AGOP after the CV
    model.agop_diag = True
    start = time.time()
    model.fit(X_train, y_train, T_train)
    print(f"fit ({time.time() - start:.1f} s): alpha = {np.exp(-model.psc):.3g}")
    if model.M is None:   # greedy AGOP: no iteration lowered the objective
        print("AGOP: M = I kept (no AGOP iteration lowered the objective)")
    else:
        print("AGOP diagonal:", np.round(model.M, 3))

    # ------------------------------------------------------------- 5. prediction
    model.do_corr = False
    p_uncorr = model.predict_proba(X_test, T_test_train)    # unconditioned prediction
    model.do_corr = True
    p_corr = model.predict_proba(X_test, T_test_train)      # with the correction of Eq. 6

    rf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]

    print("\nTest ROC AUC")
    for name, p in [("phyloK", p_corr), ("phyloK (uncorrected)", p_uncorr), ("random forest", p_rf)]:
        print(f"  {name:22s} {roc_auc_score(y_test, p):.3f}")
