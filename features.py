"""Time-series preprocessing and tabular feature construction."""
import numpy as np

def zscore_time_series(bold):
    """Z-score each subject's time series separately for every region."""
    std = bold.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (bold - bold.mean(axis=1, keepdims=True)) / std

def build_features(bold, age, sex, scanner):
    """Return z-scored series, 25-style range features, and demographics."""
    z = zscore_time_series(bold)
    range_features = z.max(axis=1) - z.min(axis=1)
    demographics = np.column_stack((age, sex, scanner))
    X = np.column_stack((range_features, demographics))
    return z, range_features, X
