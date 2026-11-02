from __future__ import annotations


import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()


def _z(alpha_two_sided: float) -> float:
    return _ND.inv_cdf(1.0 - alpha_two_sided / 2.0)


def _gammainc_upper(a: float, x: float) -> float:
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b, c, d = x + 1.0 - a, 1.0 / 1e-300, 1.0 / (x + 1.0 - a)   # continued fraction for Q
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x: float, df: int) -> float:
    return _gammainc_upper(df / 2.0, x / 2.0) if df > 0 and x == x else float("nan")


# ---------------------------------------------------------------------------
# ranks
# ---------------------------------------------------------------------------
def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# omnibus tests
# ---------------------------------------------------------------------------
def friedman(wide: np.ndarray) -> dict:
    """wide: cases x arms (complete blocks).  Friedman chi-square with the tie correction (Conover form) and Kendall's W."""
    m = np.asarray(wide, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return {"n": n, "k": k, "statistic": float("nan"), "df": k - 1, "p": float("nan"), "kendall_w": float("nan")}
    r = np.vstack([rankdata(row) for row in m])
    R = r.sum(axis=0)
    A1 = float((r ** 2).sum())
    C1 = n * k * (k + 1) ** 2 / 4.0
    den = A1 - C1
    if den <= 0:                                      # every block completely tied
        return {"n": n, "k": k, "statistic": 0.0, "df": k - 1, "p": 1.0, "kendall_w": 0.0, "mean_ranks": (R / n).tolist()}
    q = (k - 1) * (float((R ** 2).sum()) - n * C1) / den      # = 12/(nk(k+1))·ΣR² − 3n(k+1) when there are no ties
    return {"n": n, "k": k, "statistic": float(q), "df": k - 1, "p": chi2_sf(q, k - 1), "kendall_w": float(q / (n * (k - 1))),
            "mean_ranks": (R / n).round(4).tolist(), "method": "Friedman test (tie-corrected), blocks = cases"}


def cochran_q(wide: np.ndarray) -> dict:
    """wide: cases x arms of 0/1.  Cochran's Q test for equality of the arm-wise event proportions (paired)."""
    m = np.asarray(wide).astype(int)
    n, k = m.shape
    col, row, N = m.sum(axis=0), m.sum(axis=1), int(m.sum())
    den = k * N - float((row ** 2).sum())
    if den <= 0:
        return {"n": n, "k": k, "statistic": 0.0, "df": k - 1, "p": 1.0, "events": col.tolist(), "note": "no discordant blocks"}
    q = (k - 1) * (k * float((col ** 2).sum()) - N ** 2) / den
    return {"n": n, "k": k, "statistic": float(q), "df": k - 1, "p": chi2_sf(q, k - 1), "events": col.tolist(), "method": "Cochran's Q"}


# ---------------------------------------------------------------------------
# paired contrasts
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank(d: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired differences d (zeros dropped; average ranks; tie + continuity correction)."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "w_minus": 0.0, "z": 0.0, "p": 1.0, "rank_biserial": 0.0}
    r = rankdata(np.abs(d))
    wp, wm = float(r[d > 0].sum()), float(r[d < 0].sum())
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(((counts ** 3 - counts).sum())) / 48.0
    if var <= 0:
        return {"n_nonzero": n, "w_plus": wp, "w_minus": wm, "z": 0.0, "p": 1.0, "rank_biserial": (wp - wm) / (n * (n + 1) / 2.0)}
    cc = 0.5 * np.sign(wp - mean)
    z = (wp - mean - cc) / math.sqrt(var)
    return {"n_nonzero": n, "w_plus": wp, "w_minus": wm, "z": float(z), "p": float(min(1.0, 2 * (1 - _ND.cdf(abs(z))))),
            "rank_biserial": float((wp - wm) / (n * (n + 1) / 2.0))}


def signflip_p(d: np.ndarray, perms: int = 20000, seed: int = 20260920) -> float:
    """Two-sided sign-flip permutation p-value for H0: mean paired difference = 0."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(perms, len(d)))
    return float(((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / (perms + 1))


def paired_score_contrast(x: np.ndarray, y: np.ndarray, boot: int = 10000, seed: int = 20260920, alpha: float = 0.05) -> dict:
    """x, y: case-level scores of arm X and arm Y on the SAME cases.  Positive difference = X scores higher than Y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    w = wilcoxon_signed_rank(d)
    rng = np.random.default_rng(seed)
    bm = d[rng.integers(0, n, size=(boot, n))].mean(axis=1) if n else np.array([float("nan")])
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    return {"n": n, "mean_x": float(x.mean()), "mean_y": float(y.mean()), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "sd_diff": sd, "ci95": [float(np.quantile(bm, alpha / 2)), float(np.quantile(bm, 1 - alpha / 2))],
            "cohen_dz": float(d.mean() / sd) if sd and sd > 0 else 0.0, "rank_biserial": w["rank_biserial"],
            "wilcoxon_z": w["z"], "p": w["p"], "n_nonzero": w["n_nonzero"], "p_signflip": signflip_p(d, seed=seed),
            "method": "Wilcoxon signed-rank (two-sided); CI = case-level percentile bootstrap of the mean paired difference"}


# ---- paired binary endpoint -------------------------------------------------------------------------------------------
def _q_tilde(n10: int, n01: int, n: int, delta: float) -> float:
    A = 2.0 * n
    B = -(n10 + n01) + (2.0 * n - n10 + n01) * delta
    C = -n01 * delta * (1.0 - delta)
    disc = max(0.0, B * B - 4.0 * A * C)
    return (-B + math.sqrt(disc)) / (2.0 * A)


def tango_z(n10: int, n01: int, n: int, delta: float) -> float:
    q = _q_tilde(n10, n01, n, delta)
    var = n * (2.0 * q + delta - delta * delta)
    if var <= 0:
        return 0.0 if abs(n10 - n01 - n * delta) < 1e-12 else math.copysign(1e12, n10 - n01 - n * delta)
    return (n10 - n01 - n * delta) / math.sqrt(var)


def _bisect(f, lo: float, hi: float, iters: int = 200) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def tango_ci(n10: int, n01: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Two-sided (1-alpha) Tango (1998) score CI for a paired difference of proportions.  Returns (diff, lower, upper)."""
    if n <= 0:
        raise ValueError("n must be positive")
    diff = (n10 - n01) / n
    z = _z(alpha)
    eps = 1e-9
    lower = _bisect(lambda d: tango_z(n10, n01, n, d) - z, -1.0 + eps, 1.0 - eps)
    upper = _bisect(lambda d: tango_z(n10, n01, n, d) + z, -1.0 + eps, 1.0 - eps)
    return diff, max(-1.0, lower), min(1.0, upper)


def mcnemar_exact(n10: int, n01: int) -> float:
    """Two-sided exact McNemar (binomial) p-value."""
    m = n10 + n01
    if m == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
    return min(1.0, 2.0 * p)


def paired_counts(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    return int(((a == 1) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum()), \
        int(((a == 0) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())


def paired_binary_contrast(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict:
    """x, y: 0/1 events of arm X and arm Y on the same cases.  diff = rate(X) - rate(Y)."""
    n11, n10, n01, n00 = paired_counts(x, y)
    n = n11 + n10 + n01 + n00
    diff, lo, hi = tango_ci(n10, n01, n, alpha)
    return {"n": n, "n11": n11, "n10": n10, "n01": n01, "n00": n00, "rate_x": (n11 + n10) / n, "rate_y": (n11 + n01) / n,
            "diff": diff, "ci95": [lo, hi], "p": mcnemar_exact(n10, n01),
            "method": "exact McNemar (two-sided); CI = Tango score interval for the paired difference"}


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------
def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else 2.0))
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)) if p == p else running
        out[k] = running if p == p else float("nan")
    return out


# ---------------------------------------------------------------------------
# planning: paired difference
# ---------------------------------------------------------------------------
def power_paired(n: int, delta: float, sd_diff: float, alpha_two_sided: float = 0.05) -> float:
    """Normal-approximation power of a two-sided paired test for a true mean difference `delta` (same units as sd_diff)."""
    if sd_diff <= 0 or n < 2:
        return float("nan")
    ncp = abs(delta) / (sd_diff / math.sqrt(n))
    z = _z(alpha_two_sided)
    return float(_ND.cdf(ncp - z) + _ND.cdf(-ncp - z))


def mdd_paired(n: int, sd_diff: float, alpha_two_sided: float = 0.05, power: float = 0.8) -> float:
    """Minimal detectable mean paired difference at the given power."""
    return float((_z(alpha_two_sided) + _ND.inv_cdf(power)) * sd_diff / math.sqrt(n))


def n_paired(delta: float, sd_diff: float, alpha_two_sided: float = 0.05, power: float = 0.8) -> int:
    return int(math.ceil(((_z(alpha_two_sided) + _ND.inv_cdf(power)) * sd_diff / abs(delta)) ** 2))


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    po = float((a == b).mean())
    pe = float(sum(((a == c).mean() * (b == c).mean()) for c in (0, 1)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def gwet_ac1(ratings: np.ndarray) -> float:
    r = np.asarray(ratings).astype(int)
    n_items, n_raters = r.shape
    if n_raters < 2:
        return float("nan")
    k1 = r.sum(axis=1)
    pa = float((((k1 * (k1 - 1)) + ((n_raters - k1) * (n_raters - k1 - 1))) / (n_raters * (n_raters - 1))).mean())
    pi = float(r.mean())
    pe = 2 * pi * (1 - pi)
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def icc2k(m: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,k): two-way random, absolute agreement, mean of k raters.  m: targets x raters (complete)."""
    m = np.asarray(m, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = m.mean()
    msr = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def icc2_single(m: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,1): two-way random, absolute agreement, single rating.  m: targets x raters (complete)."""
    m = np.asarray(m, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = m.mean()
    msr = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def icc1(m: np.ndarray) -> dict:

    m = np.asarray(m, dtype=float)
    n, k = m.shape
    if n < 2 or k < 2:
        return {"icc1_1": float("nan"), "icc1_k": float("nan"), "n_targets": n, "k": k}
    gm = m.mean()
    msb = k * ((m.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msw = ((m - m.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return {"icc1_1": float((msb - msw) / (msb + (k - 1) * msw)) if msb + (k - 1) * msw > 0 else float("nan"),
            "icc1_k": float((msb - msw) / msb) if msb > 0 else float("nan"), "n_targets": n, "k": k}


def spearman(x, y) -> float:
    rx, ry = rankdata(np.asarray(x)), rankdata(np.asarray(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# regression helpers (case-cluster robust), used for adjusted sensitivity models
# ---------------------------------------------------------------------------
def logit_cluster_robust(y: np.ndarray, X: np.ndarray, groups: np.ndarray, iters: int = 100) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    W = mu * (1 - mu) + 1e-9
    bread = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]))
    meat = np.zeros_like(bread)
    ug = np.unique(groups)
    for g in ug:
        idx = groups == g
        s = X[idx].T @ (y[idx] - mu[idx])
        meat += np.outer(s, s)
    G = len(ug)
    cov = bread @ meat @ bread * (G / max(G - 1, 1))
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = np.where(se > 0, beta / np.where(se > 0, se, 1), np.nan)
    pval = np.array([2 * (1 - _ND.cdf(abs(v))) if np.isfinite(v) else np.nan for v in z])
    return {"coef": beta, "se": se, "z": z, "p": pval, "cov": cov, "n_clusters": G}


def ols_cluster_robust(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    y, X = np.asarray(y, float), np.asarray(X, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    res = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(groups)
    for g in ug:
        i = groups == g
        s = X[i].T @ res[i]
        meat += np.outer(s, s)
    G, n, p = len(ug), len(y), X.shape[1]
    cov = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return {"coef": beta, "se": se, "cov": cov, "n_clusters": G}


def wald_contrast(fit: dict, c: np.ndarray) -> dict:
    """Linear contrast c'beta from a cluster-robust fit: estimate, SE, 95% CI, two-sided p (normal reference)."""
    c = np.asarray(c, float)
    est = float(c @ fit["coef"])
    se = float(math.sqrt(max(0.0, c @ fit["cov"] @ c)))
    p = float(2 * (1 - _ND.cdf(abs(est / se)))) if se > 0 else float("nan")
    return {"estimate": est, "se": se, "ci95": [est - 1.959964 * se, est + 1.959964 * se], "p": p}


