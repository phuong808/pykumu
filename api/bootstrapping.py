"""Bootstrapping configuration helpers."""

from __future__ import annotations


def configure_bootstrapping(search, algorithm: str, n_bootstrap: int) -> None:
    """Configure algorithm-specific bootstrapping settings."""
    if algorithm == "boss":
        search.set_bootstrapping(
            numberResampling=n_bootstrap,
            percent_resample_size=100,
            seed=32,
            add_original=True,
            with_replacement=True,
            resampling_ensemble=1,
        )
    else:
        search.set_bootstrapping(
            numberResampling=n_bootstrap,
            percent_resample_size=90,
            seed=32,
            add_original=True,
            with_replacement=True,
            resampling_ensemble=1,
        )
