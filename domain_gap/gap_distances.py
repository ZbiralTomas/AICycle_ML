"""
Domain-gap distances between two sets of feature/embedding vectors X, Y.

  mmd_rbf        : headline statistic; unbiased MMD^2 with RBF kernel
                   (median-heuristic bandwidth) + permutation p-value
  energy_distance: parameter-free distributional distance
  kid            : Kernel Inception Distance style (degree-3 poly kernel MMD),
                   unbiased -- preferred over FID for small N
  proxy_a_distance: domain-classifier separability (CV ROC-AUC -> PAD)
  coral          : Frobenius distance between feature covariances
  hue_emd        : 1-D Wasserstein between pooled hue values (companion)

All operate on already-standardized inputs where noted.
"""
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def _pair_sqdists(A, B):
    return cdist(A, B, "sqeuclidean")


def _median_bandwidth(X, Y):
    Z = np.vstack([X, Y])
    n = min(len(Z), 500)
    idx = np.random.RandomState(0).choice(len(Z), n, replace=False)
    d = _pair_sqdists(Z[idx], Z[idx])
    med = np.median(d[d > 0])
    return med if med > 0 else 1.0


def _mmd2_rbf(X, Y, gamma):
    Kxx = np.exp(-gamma * _pair_sqdists(X, X))
    Kyy = np.exp(-gamma * _pair_sqdists(Y, Y))
    Kxy = np.exp(-gamma * _pair_sqdists(X, Y))
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())


def mmd_rbf(X, Y, n_perm=500, seed=0):
    """Unbiased MMD^2 (RBF) + permutation p-value."""
    med = _median_bandwidth(X, Y)
    gamma = 1.0 / med
    obs = _mmd2_rbf(X, Y, gamma)
    if n_perm:
        rng = np.random.RandomState(seed)
        Z = np.vstack([X, Y])
        n = len(X)
        cnt = 0
        for _ in range(n_perm):
            perm = rng.permutation(len(Z))
            if _mmd2_rbf(Z[perm[:n]], Z[perm[n:]], gamma) >= obs:
                cnt += 1
        p = (cnt + 1) / (n_perm + 1)
    else:
        p = np.nan
    return float(obs), float(p)


def energy_distance(X, Y):
    dxy = cdist(X, Y).mean()
    dxx = cdist(X, X).mean()
    dyy = cdist(Y, Y).mean()
    return float(2 * dxy - dxx - dyy)


def kid(X, Y):
    """Unbiased KID (degree-3 polynomial kernel MMD)."""
    d = X.shape[1]
    k = lambda A, B: (A @ B.T / d + 1.0) ** 3
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                 - 2.0 * Kxy.mean())


def proxy_a_distance(X, Y, seed=0):
    """Domain-classifier separability. Returns (auc, PAD)."""
    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = np.zeros(len(Z))
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Z[tr], y[tr])
        scores[te] = clf.predict_proba(Z[te])[:, 1]
    auc = roc_auc_score(y, scores)
    err = 1.0 - max(auc, 1 - auc)      # balanced-error proxy
    pad = 2.0 * (1.0 - 2.0 * err)
    return float(auc), float(pad)


def coral(X, Y):
    cx = np.cov(X, rowvar=False)
    cy = np.cov(Y, rowvar=False)
    return float(np.linalg.norm(cx - cy, "fro"))


def hue_emd(hue_x, hue_y):
    return float(wasserstein_distance(hue_x, hue_y))


# --------------------------- shared driver -----------------------------------
def standardize(X, Y):
    """z-score both sets using the pooled mean/std (per comparison)."""
    Z = np.vstack([X, Y])
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    return (X - mu) / sd, (Y - mu) / sd


def compare(X, Y, n_perm=500, B=300, seed=0):
    """
    All distances on equal-N, standardized feature/embedding matrices, with a
    bootstrap CI on the headline MMD. Used by BOTH the statistical and latent
    stages so the two are methodologically identical apart from the features.
    """
    n = min(len(X), len(Y))
    rng = np.random.RandomState(seed)
    Xs, Ys = standardize(X[rng.choice(len(X), n, replace=False)],
                         Y[rng.choice(len(Y), n, replace=False)])
    res = {}
    res["mmd"], res["mmd_p"] = mmd_rbf(Xs, Ys, n_perm=n_perm, seed=seed)
    res["energy"] = energy_distance(Xs, Ys)
    res["kid"] = kid(Xs, Ys)
    res["auc"], res["pad"] = proxy_a_distance(Xs, Ys, seed=seed)
    res["coral"] = coral(Xs, Ys)
    boot = [mmd_rbf(Xs[rng.randint(0, n, n)], Ys[rng.randint(0, n, n)],
                    n_perm=0)[0] for _ in range(B)]
    res["mmd_ci"] = [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))]
    return res
