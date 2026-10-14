"""Data loading and reproducible synthetic fMRI-like data generation."""
import numpy as np

SEED = 42

def make_synthetic(n=715, n_heavy=122, T=269, R=25, seed=SEED):
    """Generate synthetic time-series data for pipeline testing (not real subjects)."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(12, 22, n)
    sex = rng.integers(0, 2, n)
    scanner = (rng.random(n) < 0.4).astype(int)
    p = 1 / (1 + np.exp(-0.55 * (age - 17)))
    p /= p.sum()
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_heavy, replace=False, p=p)] = 1
    noise = rng.standard_normal((n, T, R))
    bold = np.zeros_like(noise)
    for t in range(1, T):
        bold[:, t] = 0.6 * bold[:, t - 1] + noise[:, t]
    bold *= rng.uniform(0.8, 1.5, (n, 1, R))
    return {"bold": bold, "age": age, "sex": sex, "scanner": scanner, "y": y}

def load_data(use_real=False, path="ncanda_rsfmri.npz", seed=SEED):
    """Load the expected real-data NPZ or fall back to synthetic data."""
    if use_real:
        with np.load(path) as f:
            return {"bold": f["bold"], "age": f["age"], "sex": f["sex"],
                    "scanner": f["scanner"], "y": f["heavy"]}
    return make_synthetic(seed=seed)
