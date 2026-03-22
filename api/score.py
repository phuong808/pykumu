"""Score-configuration helpers for Tetrad search runs."""

from __future__ import annotations


def configure_default_score(
    search,
    penalty_discount: float = 2,
    sem_bic_rule: int = 1,
    structure_prior: float = 0,
    singularity_lambda: float = 0.0,
) -> None:
    """Configure the SEM-BIC score used by the notebook workflow."""
    search.use_sem_bic(
        penalty_discount=penalty_discount,
        sem_bic_rule=sem_bic_rule,
        structurePrior=structure_prior,
        singularity_lambda=singularity_lambda,
    )
