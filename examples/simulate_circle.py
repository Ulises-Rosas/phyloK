"""
Simulate the circle dataset used by circle_final.py and save it to circle_data.npz.

The shipped circle_data.npz was made with this script. Rerunning it can give a
slightly different dataset on another machine: rng.multivariate_normal uses a
matrix decomposition whose result depends on the NumPy version and the linear
algebra library. circle_final.py therefore loads the saved data.

Run from this folder:
    python simulate_circle.py
"""
import numpy as np
from phylokrr.treeio import parseNewickTree, is_leaf, get_adj, get_C, get_APSP, get_BM
from phyloK.alpha_beta_weighting import alpha_weighting, split_data_idx

seed = 12038
rng = np.random.default_rng(seed)

# ------------------------------------------------------------------ tree
with open("test_tree_v5.txt") as f:
    newick = f.readline().strip()

nodes, root = parseNewickTree(newick)
leaves = [n for n in nodes if is_leaf(n)]
names = [n.name for n in leaves]                  # row order of X and y
adj = get_adj(nodes)
C = get_C(nodes, adj)
D = get_APSP(leaves, adj, C)                      # path lengths between tips
n = len(leaves)

# ------------------------------------------------------ circle data
# predictors and errors evolve under Brownian motion on the tree (c1 = c2 = 1)
sigma = 2 / np.max(D)                             # 2 * alpha * c, with alpha = 1 / t_max
V = sigma * get_BM(D, root, leaves, adj, C)
X = rng.multivariate_normal(np.zeros(n), V, 2).T  # n x 2
e = rng.multivariate_normal(np.zeros(n), V, 1).ravel()

f = 1.25 - np.sqrt(X[:, 0]**2 + 1.5 * X[:, 1]**2) + 0.1   # circle
y = (1 / (1 + np.exp(-(f + e))) >= 0.5).astype(float)     # y = 1 if Logistic(f + e) >= 1/2

# ---------------------------------- phylogeny-informed split (rho = 5)
A = alpha_weighting(newick, alpha=5.0, tol=1e-15, branch_length=False)
P = np.array([A[name] for name in names])
test_idx, train_idx = split_data_idx(int(0.5 * n), P, rng)

np.savez_compressed("circle_data.npz", X=X, y=y, names=np.array(names),
                    train_idx=np.array(train_idx), test_idx=np.array(test_idx))
print(f"saved circle_data.npz: {n} taxa, {len(train_idx)} training, {len(test_idx)} test")
