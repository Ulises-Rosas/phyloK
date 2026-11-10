# phyloK

Non-parametric phylogenetic regression with tree-biased kernels. phyloK kernelizes the penalized quasi-likelihood and adds the phylogeny to the kernel,

$$k(\mathbf x_i, \mathbf x_j) = \exp\left(-\gamma(\mathbf x_i-\mathbf x_j)^\top\mathbf M(\mathbf x_i-\mathbf x_j)\right) + \exp\left( -\alpha t_{ij} \right) $$

where $t_{ij}$ is the path length between taxa $i$ and $j$, $\alpha$ is learned, and $\mathbf M$ is learned with the average gradient outer product (AGOP). Predictions for new taxa can be corrected using their phylogenetic correlation with the training taxa.

Models:

| Class | Response |
|---|---|
| `phyloK.phyloK_v2.Logistic` | binary |
| `phyloK.phyloK_v2.Multinomial` | multiclass (softmax) |
| `phyloK.phyloK_v2.PhyloKernel` | continuous |

## Installation

```bash
pip install phylok
```

or, from this folder, `pip install .`. Dependencies: `numpy`, `scipy`, `joblib`, and  `phylokrr`. The example below also uses `scikit-learn`.

## Example: the circle function

[`examples/circle_final.py`](examples/circle_final.py) test a binary trait on the 500-taxon tree in `examples/test_tree_v5.txt` and fits the default ("final") phyloK model. The model uses the OU tree-biased kernel, the correction during cross-validation, residual scaling, and diagonal AGOP.

```bash
cd examples
python circle_final.py
```

### Inputs

phyloK takes standardized predictors `X`, responses `y`, and the rescaled path lengths between taxa, `T = 2 D / max(D)`:
- `T_train` is training × training.
- `T_test_train` is test × training.

In the example:
- The two predictors evolve under Brownian motion.
- `y = 1` if $\text{Logistic}(1.35 - \sqrt{\mathbf{x}_1 + 1.5 \mathbf{x}_2^2} + \varepsilon) \geq 0.5$.
- Half of the taxa are held out with a phylogeny-informed split ($\rho = 2$), so the test taxa come from clades with few training relatives.

The correction needs $\mathbf R_{nn}^{-1}\mathbf e$. If the model is given the training tree, this product is computed by recursions on the tree in $O(n)$, instead of a dense $O(n^3)$ solve. The branch lengths must be on the same scale as `T`, so they are divided by the tree height, `max(D) / 2`:

```python
from phyloK.tree_solve import get_training_tree

tree_nodes, _ = parseNewickTree(newick)
for node in tree_nodes:
    node.branch_length /= np.max(D) / 2
spps_train = np.array(names)[train_idx]                 # row order of X_train
tree_train = get_training_tree(tree_nodes, spps_train)  # pruned to the training taxa: (nodes, root)
```

### Fitting

```python
from phyloK.block_cross_validation import k_fold_cv_weigthed
from phyloK.phyloK_v2 import Logistic

model = Logistic(tol=1e-6, max_iter=100, kernel='rbf', verbose=False)
model.memory_qn = len(y_train)

# 100 log-uniform hyperparameter samples, five folds
params = {'lam':   np.logspace(-12, 3, 100, base=2),
          'gamma': np.logspace(-15, 4, 100, base=2),
          'corr':  np.logspace(-10, 0, 100, base=2)}
W_train = np.full(len(y_train), 1 / len(y_train))
best = k_fold_cv_weigthed(X_train, y_train, T_train, model, W_train, params,
                          folds=5, sample=100, seed=12038, cores=5,
                          nodes=tree_train[0], leaves=spps_train)  # pruned again in each fold

model.set_params(**best)
model.set_training_tree(tree_train, spps_train)   # tree-based solve in the fit and the prediction
model.do_agop, model.agop_diag = True, True   # AGOP after the CV
model.fit(X_train, y_train, T_train)
```

### Prediction

```python
model.do_corr = False
p_uncorrected = model.predict_proba(X_test, T_test_train)
model.do_corr = True
p = model.predict_proba(X_test, T_test_train)   # corrected with the training responses
```

### Output

The cross-validation took about 10 s on 5 cores.

```
500 taxa: 250 training, 250 test (class 1: 0.62 train, 0.54 test)
CV (6 s), best hyperparameters: {'lam': 0.000413, 'gamma': 3.7, 'corr': 0.705}
fit (0.8 s): alpha = 2.74
AGOP: M = I kept (no AGOP iteration lowered the objective)

Test ROC AUC
  phyloK                 0.886
  phyloK (uncorrected)   0.876
  random forest          0.776
```

Notes:

- During the cross-validation, some hyperparameter samples print `No convergence for weights after 100 iterations`. This is expected for extreme values of `lam` and `gamma`; those samples simply score poorly.
- AGOP is greedy. When no AGOP iteration lowers the objective, the fit without AGOP is kept (`model.M is None`). Here both predictors are relevant, so reweighting them does not help.
- In `predict_proba`, `do_corr` switches the correction of predictions on or off.
- The tree-based solve gives the same predictions as the dense solve (up to about 1e-13 here); it only changes the cost.
