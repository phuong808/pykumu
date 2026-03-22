"""Algorithm execution helpers for structure learning."""

from __future__ import annotations

import time

import api.translate as tr


def _load_java_modules():
    """Load Java-backed Tetrad modules after the JVM is initialized."""
    import edu.cmu.tetrad.algcomparison.algorithm.oracle.cpdag as cpdag
    import edu.cmu.tetrad.algcomparison.score as score_
    import edu.cmu.tetrad.data as td
    import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp
    import java.io as io
    from edu.cmu.tetrad.util import Params, Parameters

    return cpdag, score_, td, gp, io, Params, Parameters


def _build_common_state(
    data,
    *,
    n_bootstrap: int,
    knowledge_file: str | None,
    penalty_discount: float,
    sem_bic_rule: int,
    structure_prior: float,
    singularity_lambda: float,
    bootstrap_percent_resample_size: int,
    bootstrap_seed: int,
    bootstrap_add_original: bool,
    bootstrap_with_replacement: bool,
    bootstrap_resampling_ensemble: int,
    verbose: bool,
):
    """Build shared Tetrad state for FGES and BOSS runs."""
    _, score_, td, _, io, Params, Parameters = _load_java_modules()
    tetrad_data = tr.pandas_data_to_tetrad(data)
    params = Parameters()

    params.set(Params.PENALTY_DISCOUNT, penalty_discount)
    params.set(Params.SEM_BIC_STRUCTURE_PRIOR, structure_prior)
    params.set(Params.SEM_BIC_RULE, sem_bic_rule)
    params.set(Params.SINGULARITY_LAMBDA, singularity_lambda)
    score = score_.SemBicScore()

    params.set(Params.NUMBER_RESAMPLING, n_bootstrap)
    params.set(Params.PERCENT_RESAMPLE_SIZE, bootstrap_percent_resample_size)
    params.set(Params.ADD_ORIGINAL_DATASET, bootstrap_add_original)
    params.set(Params.RESAMPLING_WITH_REPLACEMENT, bootstrap_with_replacement)
    params.set(Params.RESAMPLING_ENSEMBLE, bootstrap_resampling_ensemble)
    params.set(Params.SEED, bootstrap_seed)
    params.set(Params.VERBOSE, verbose)

    if knowledge_file:
        know_file = io.File(knowledge_file)
        know_delim = td.DelimiterType.WHITESPACE
        knowledge = td.SimpleDataLoader.loadKnowledge(know_file, know_delim, "#")
    else:
        knowledge = td.Knowledge()

    return tetrad_data, params, score, knowledge


def run_fges_algorithm(
    data,
    n_bootstrap: int,
    knowledge_file: str | None = None,
    penalty_discount: float = 2,
    sem_bic_rule: int = 1,
    structure_prior: float = 0,
    singularity_lambda: float = 0.0,
    bootstrap_percent_resample_size: int = 90,
    bootstrap_seed: int = 32,
    bootstrap_add_original: bool = True,
    bootstrap_with_replacement: bool = True,
    bootstrap_resampling_ensemble: int = 1,
    verbose: bool = False,
    max_degree: int = 1000,
    faithfulness_assumed: bool = True,
    symmetric_first_step: bool = True,
    parallelized: bool = False,
):
    """Run FGES directly from a pandas DataFrame and return elapsed, graph, and JSON."""
    cpdag, _, _, gp, _, Params, _ = _load_java_modules()
    tetrad_data, params, score, knowledge = _build_common_state(
        data,
        n_bootstrap=n_bootstrap,
        knowledge_file=knowledge_file,
        penalty_discount=penalty_discount,
        sem_bic_rule=sem_bic_rule,
        structure_prior=structure_prior,
        singularity_lambda=singularity_lambda,
        bootstrap_percent_resample_size=bootstrap_percent_resample_size,
        bootstrap_seed=bootstrap_seed,
        bootstrap_add_original=bootstrap_add_original,
        bootstrap_with_replacement=bootstrap_with_replacement,
        bootstrap_resampling_ensemble=bootstrap_resampling_ensemble,
        verbose=verbose,
    )

    params.set(Params.SYMMETRIC_FIRST_STEP, symmetric_first_step)
    params.set(Params.MAX_DEGREE, max_degree)
    params.set(Params.PARALLELIZED, parallelized)
    params.set(Params.FAITHFULNESS_ASSUMED, faithfulness_assumed)

    alg = cpdag.Fges(score)
    alg.setKnowledge(knowledge)

    start_time = time.time()
    graph = alg.search(tetrad_data, params)
    elapsed = time.time() - start_time
    graph_json = str(gp.graphToJson(graph))
    return elapsed, graph, graph_json


def run_boss_algorithm(
    data,
    n_bootstrap: int,
    knowledge_file: str | None = None,
    penalty_discount: float = 2,
    sem_bic_rule: int = 1,
    structure_prior: float = 0,
    singularity_lambda: float = 0.0,
    bootstrap_percent_resample_size: int = 100,
    bootstrap_seed: int = 32,
    bootstrap_add_original: bool = True,
    bootstrap_with_replacement: bool = True,
    bootstrap_resampling_ensemble: int = 1,
    verbose: bool = False,
    num_starts: int = 1,
    use_bes: bool = False,
    time_lag: int = 0,
    use_data_order: bool = True,
    output_cpdag: bool = True,
):
    """Run BOSS directly from a pandas DataFrame and return elapsed, graph, and JSON."""
    cpdag, _, _, gp, _, Params, _ = _load_java_modules()
    tetrad_data, params, score, knowledge = _build_common_state(
        data,
        n_bootstrap=n_bootstrap,
        knowledge_file=knowledge_file,
        penalty_discount=penalty_discount,
        sem_bic_rule=sem_bic_rule,
        structure_prior=structure_prior,
        singularity_lambda=singularity_lambda,
        bootstrap_percent_resample_size=bootstrap_percent_resample_size,
        bootstrap_seed=bootstrap_seed,
        bootstrap_add_original=bootstrap_add_original,
        bootstrap_with_replacement=bootstrap_with_replacement,
        bootstrap_resampling_ensemble=bootstrap_resampling_ensemble,
        verbose=verbose,
    )

    params.set(Params.USE_BES, use_bes)
    params.set(Params.NUM_STARTS, num_starts)
    params.set(Params.TIME_LAG, time_lag)
    params.set(Params.USE_DATA_ORDER, use_data_order)
    params.set(Params.OUTPUT_CPDAG, output_cpdag)

    alg = cpdag.Boss(score)
    alg.setKnowledge(knowledge)

    start_time = time.time()
    graph = alg.search(tetrad_data, params)
    elapsed = time.time() - start_time
    graph_json = str(gp.graphToJson(graph))
    return elapsed, graph, graph_json
