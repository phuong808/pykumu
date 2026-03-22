"""Data preparation and transformation helpers for causal analysis workflows."""

from __future__ import annotations

from statistics import NormalDist
import numpy as np
import pandas as pd


def add_time_lag_by_group(
    dt: pd.DataFrame,
    group_col: str,
    time_col: str,
    lag_feature_cols: list[str],
    lag_suffix: str = "2",
) -> pd.DataFrame:
    """Add one-step lag features by group and drop terminal rows without a next step."""
    dt = dt.sort_values([group_col, time_col]).reset_index(drop=True)

    def _add_group_lag(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) < 2:
            lag_cols = pd.DataFrame(
                {f"{col}{lag_suffix}": np.nan for col in lag_feature_cols},
                index=group.index,
            )
            return pd.concat([group, lag_cols], axis=1)

        lag_cols = pd.DataFrame(
            {f"{col}{lag_suffix}": group[col].shift(-1).values for col in lag_feature_cols},
            index=group.index,
        )
        return pd.concat([group, lag_cols], axis=1).iloc[:-1].copy()

    parts = [_add_group_lag(group) for _, group in dt.groupby(group_col, sort=False)]
    return pd.concat(parts, ignore_index=True)


def remove_short_groups(df: pd.DataFrame, group_col: str, min_rows: int = 8):
    """Remove groups with fewer than min_rows observations."""
    counts = df.groupby(group_col).size().reset_index(name="n_rows").sort_values("n_rows")
    short = counts[counts["n_rows"] < min_rows]
    short_ids = short[group_col].values
    filtered = df[~df[group_col].isin(short_ids)].copy()
    return filtered, short


def make_null_features(df: pd.DataFrame, seed: int = 32, prefix: str = "nv-") -> pd.DataFrame:
    """Create shuffled null-feature copies of all columns and append them."""
    rng = np.random.default_rng(seed)
    null_df = pd.DataFrame({col: rng.permutation(df[col].to_numpy()) for col in df.columns})
    null_df.columns = [f"{prefix}{col}" for col in null_df.columns]
    return pd.concat([df, null_df], axis=1)


def keep_first_columns(df: pd.DataFrame, max_cols: int) -> pd.DataFrame:
    """Keep only the first max_cols columns."""
    return df.iloc[:, :max_cols].copy()


def npn_transform(df: pd.DataFrame, protected_cols=("cve_id", "start")) -> pd.DataFrame:
    """Apply an NPN-style rank Gaussianization transform to non-protected columns."""
    out = df.copy()
    normal = NormalDist()

    cols_to_transform = [c for c in out.columns if c not in set(protected_cols)]

    for col in cols_to_transform:
        x = pd.to_numeric(out[col], errors="coerce")
        valid = x.notna()

        if valid.sum() <= 1:
            continue

        ranks = x[valid].rank(method="average")
        n = float(valid.sum())
        scaled = (ranks / (n + 1.0)).clip(1e-12, 1 - 1e-12)
        out.loc[valid, col] = scaled.map(normal.inv_cdf).astype(float)

    return out
