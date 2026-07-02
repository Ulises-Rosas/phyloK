
import numpy as np
from phylokrr.treeio import parseNewickTree, is_leaf

def split_leaves(sorted_leaves, folds):

    n = len(sorted_leaves)
    fold_size = n//folds
    myfolds = []
    for i in range(folds):
        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))
        test_l = sorted_leaves[test_idx] 
        train_l = sorted_leaves[train_idx]
        myfolds.append([train_l, test_l])

    return myfolds

def tree_postorder(root):

    L = []
    S = [root]
    while S:
        v = S.pop()
        L.append(v)

        if is_leaf(v):
            continue   

        l = v.left
        r = v.right
        S.append(l)
        S.append(r)

    return L

# tree = "((4,5),(6,7));"
def alpha_weighting(tree, alpha=1/4):
    """
    Weight the leaves of a phylogenetic tree based on 
    on the number of decendents using an alpha parameter. 

    Since it uses a hash table to store the weights, 
    the order of the leaves is not guaranteed 
    to be the same as in the input tree.

    """
    _, root = parseNewickTree(tree)

    L = tree_postorder(root)

    while L:
        v = L.pop()
        if is_leaf(v):
            v.branch_length = 1
            continue

        l = v.left
        r = v.right
        v.branch_length = l.branch_length + r.branch_length

    # alpha = 1/4
    A = {}
    # S = [(root, math.log(1))]
    S = [(root, 1)]
    while S:
        pa,p = S.pop()

        if is_leaf(pa):
            # store only leaves probabilities
            A[pa.name] = p
            continue

        nl = pa.left.branch_length
        nr = pa.right.branch_length
        
        pr = (nr/(nl+nr))**alpha
        pl = 1 - pr
        # S.append((pa.left, math.log(pl) + p))
        # S.append((pa.right, math.log(pr) + p))
        S.append((pa.left, pl * p))
        S.append((pa.right, pr * p))

    return A

def sample_mChoice(P, rng, Fz):
    """

    Return a set of indices sampled according to P, without replacement, and with size Fz
    """
    S_idx = rng.choice(len(P), size=Fz, p=P, replace=False)
    P[S_idx] = 0
    P /= np.sum(P)

    return set(S_idx)

def split_data_idx(num_test, P, rng): # type: ignore
    """

    Return two lists of indices, one for test and one for train
    """
    all_idx = set(range(len(P)))
    test_idx = sample_mChoice(P, rng, num_test)
    return list(test_idx), list(all_idx - test_idx)

def getT2(P, rng, n, folds = 5):
    """
    Return a list of sets
    """
    T = []
    Fz = n//folds
    # O(k * n)
    for _ in range(folds - 1):
        T.append(
            sample_mChoice(P, rng, Fz)
        )

    last_S = []
    for i in range(n):
        if P[i] != 0.0:
            last_S.append(i)
    T.append(set(last_S))

    return T


def weighted_sample(P, rng, folds=5): # type: ignore 

    n = len(P)
    all_idx = set(range(n))
    T = getT2(P, rng, n, folds)

    myfolds = []
    for test_idx in T:
        myfolds.append([
            list(all_idx - test_idx), 
            list(test_idx)
        ])
    return myfolds

# region: Testing

# tree_file = "test_tree_v5.txt"
# with open(tree_file, 'r') as f: 
#     tree = f.read().strip()

# from phylokrr.treeio import get_vcv

# rng = np.random.default_rng(seed=12038)
# V, leaves = get_vcv(tree,  process = "OU", sigma2 = 1, alpha = 1)

# # P = np.linalg.inv(V) @ np.ones(len(leaves))
# # P /= np.sum(P)




# A = alpha_weighting(tree, alpha = 1e2)
# P = np.array([A[leaf] for leaf in leaves])
# # print(A)
# print(P)

# n = len(leaves)
# test_idx, train_idx = split_data_idx(num_test=int(n*0.25), P=P, rng=rng)

# P_train = P[train_idx]
# myfolds = weighted_sample(P_train, rng, folds=5)

# # for a,b in myfolds:
# #     print(a,"-------",b)
# # print(P[list(myfolds[3][1])])

# test_leaves = [leaves[i] for i in test_idx]
# with open("test_leaves.txt", 'w') as f:
#     f.write(",".join(test_leaves) + "\n")
# # print("Test Leaves: ", test_leaves)

# # # # # myfolds
# with open("folds.txt", 'w') as f:
# # with open("folds_random.txt", 'w') as f:
#     for train_idx_f, test_idx_f in myfolds:

#         train_leaves_f = []
#         for idx in train_idx_f:
#             train_leaves_f.append(leaves[train_idx[idx]])

#         test_leaves_f = []
#         for idx in test_idx_f:
#             test_leaves_f.append(leaves[train_idx[idx]])

#         f.write(",".join(train_leaves_f) + "\n")
#         f.write(",".join(test_leaves_f) + "\n")

# endregion


# region: Old code

# def updateP(S, P):
#     # mark those taken as 0
#     P[S != 0] = 0
#     # P[S] = 0
#     # adjust the probabilities
#     P /= np.sum(P)
#     # print("Updated P: ", P)


# def multi_sample(P, rng, Fz):
#     # P = P.copy()
#     # S_idx = rng.choice(len(P), size=Fz, p=P, replace=False)
#     # S = np.zeros(len(P), dtype=int)
#     # S[S_idx] = 1

#     S = rng.multinomial(n=Fz, pvals=P, size=1)[0]
#     updateP(S, P)
#     NZ = sum(S != 0)
#     while Fz > NZ:
#         Si = rng.multinomial(n=Fz - NZ, pvals=P, size=1)[0]
#         # Just update with the recently taken 
#         # samples, not the whole S
#         updateP(Si, P)
#         S += Si
#         NZ = sum(S != 0)
    
#     return S

# def getT(P, rng, n, folds = 5):
#     # P = P.copy()
#     T = []  
#     Fz = n//folds
#     # O(k * n)
#     for _ in range(folds - 1):
#         S = multi_sample2(P, rng, Fz)
#         T.append(S)

#     S = np.zeros(n, dtype=int)
#     S[P != 0] = 1
#     T.append(S)
#     return T

# def split_sample(t):
#     test_idx = []
#     train_idx = []
#     for k, v in enumerate(t):
#         if v != 0:
#             test_idx.append(k)
#         else:
#             train_idx.append(k)

#     return test_idx, train_idx

# endregion