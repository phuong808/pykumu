# Bootstrap Resampling Hangs Indefinitely for `number_resampling > 50` After `tetrad-current.jar` Update

## Summary

After updating `tetrad-current.jar` to resolve a missing `GraphSaveLoadUtils.graphToJson()` method issue, both the FGES and BOSS algorithms now hang indefinitely when `number_resampling` is set above ~50. Previously (with the January 2026 jar), FGES ran successfully with up to 500 resampling iterations and BOSS with up to 1,000.

## Environment

| Component | Version |
|---|---|
| **Python** | 3.13.0 |
| **JPype1** | 1.6.0 |
| **Java** | OpenJDK 21.0.10 (Corretto-21.0.10.7.1) |
| **Tetrad JAR** | `7.6.11-SNAPSHOT` (Implementation-Version from MANIFEST.MF) |
| **Build JDK** | JDK 25 (Build-Jdk-Spec from MANIFEST.MF) |
| **OS tested** | macOS (Darwin 24.6.0) and Windows |

## Observed Behavior

- `number_resampling <= 50`: Both FGES and BOSS complete successfully and return results.
- `number_resampling > 50`: The JVM thread running the search algorithm **hangs indefinitely** — no error, no exception, no output. The process must be manually killed.

This behavior is **consistent across both macOS and Windows**.

## Expected Behavior

Based on prior working behavior with the January 2026 jar:
- **FGES**: Should complete with `number_resampling` up to at least **500**.
- **BOSS**: Should complete with `number_resampling` up to at least **1,000**.

## Timeline of Events

### 1. Working state (~January 30, 2026)

Everything worked as expected. High resampling counts completed successfully.

- **JAR file**: `tetrad-current.jar` committed at `31c01e9` (2026-01-30)
- **JAR MD5**: `42a99f5a612ad16f5b4575dc85f1c8dd`
- **JAR size**: 50,209,103 bytes
- **Build date of `Fges.class`**: 2026-01-30
- **Build date of `AbstractBootstrapAlgorithm.class`**: 2026-01-30
- **`GraphSaveLoadUtils.class` size**: 40,253 bytes

### 2. Broken `get_json` / `graphToJson` (~March 21, 2026)

A jar update introduced a regression where `edu.cmu.tetrad.graph.GraphSaveLoadUtils.graphToJson()` became inaccessible via JPype, causing `graph.get_json()` to fail.

- **JAR file**: `tetrad-current.jar` committed at `8e2a038` (2026-03-21)
- **JAR MD5**: `310f130bd7be273a33905c291aea7bf3`
- **JAR size**: 53,049,153 bytes
- **`GraphSaveLoadUtils.class` size**: 40,582 bytes (changed from 40,253)

### 3. `get_json` fixed, resampling regression introduced (~March 23, 2026)

A subsequent jar update resolved the `graphToJson` accessibility issue, but introduced the bootstrap resampling hang.

- **JAR file**: `tetrad-current.jar` committed at `00e200e` (2026-03-23)
- **JAR MD5**: `a3b57483751b8a235004e23dacdfb7a4`
- **JAR size**: 53,029,188 bytes

### 4. Current state (March 27, 2026 — latest)

The current jar still exhibits the resampling hang. `get_json` works correctly.

- **JAR file**: `tetrad-current.jar` at HEAD (`bea3e8a`, 2026-03-27)
- **JAR MD5**: `307b90e7db976c3e5566a3167c278868`
- **JAR size**: 50,764,069 bytes
- **Build date of `Fges.class`**: 2026-02-21
- **Build date of `AbstractBootstrapAlgorithm.class`**: 2026-02-21

## Steps to Reproduce

### 1. Setup

```bash
git clone https://github.com/sailuh/pykumu.git
cd pykumu
conda env create -f env.yml
conda activate pykumu-causal-analysis
pip install -e .
```

Requires Java JDK 21+ (Amazon Corretto 21 recommended) with `JAVA_HOME` set.

### 2. Minimal reproduction code

```python
import jpype
import jpype.imports

jar_path = "vignettes/resources/tetrad-current.jar"

if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[jar_path])

from pykumu import data, score, bootstrapping, algorithm, graph, knowledge

# Load data
state = data.load_continuous("vignettes/resources/null_variable_dt.csv")

# Configure score
sem_bic = score.use_sem_bic(state["params"], penalty_discount=2)

# --- THIS WORKS (number_resampling=50) ---
bootstrapping.set_bootstrapping(
    state["params"],
    number_resampling=50,
    percent_resample_size=90,
    seed=32,
    add_original_dataset=True,
    resampling_with_replacement=True,
    resampling_ensemble=1
)
result = algorithm.run_fges(
    state["data"], state["params"], sem_bic, state["knowledge"],
    symmetric_first_step=True, max_degree=1000,
    faithfulness_assumed=True, parallelized=False
)
print("50 resamplings completed successfully")

# --- THIS HANGS (number_resampling=100) ---
bootstrapping.set_bootstrapping(
    state["params"],
    number_resampling=100,       # <-- hangs at values > ~50
    percent_resample_size=90,
    seed=32,
    add_original_dataset=True,
    resampling_with_replacement=True,
    resampling_ensemble=1
)
result = algorithm.run_fges(
    state["data"], state["params"], sem_bic, state["knowledge"],
    symmetric_first_step=True, max_degree=1000,
    faithfulness_assumed=True, parallelized=False
)
print("100 resamplings completed")  # Never reached
```

### 3. Dataset details

- **File**: `vignettes/resources/null_variable_dt.csv`
- **Rows**: 4,870
- **Columns**: 138

## Relevant Code Paths in Tetrad

The bootstrap resampling is orchestrated by `AbstractBootstrapAlgorithm`, which is the superclass for both `Fges` and `Boss` in the `edu.cmu.tetrad.algcomparison.algorithm` package. The call chain is:

1. `alg.search(data, params)` — initiates the search with bootstrap parameters
2. `Params.NUMBER_RESAMPLING` — controls the iteration count
3. `alg.getBootstrapGraphs()` — retrieves results after search completes

The `AbstractBootstrapAlgorithm.class` file size is identical (6,917 bytes) across all jar versions, suggesting the regression may be in a dependency called by the bootstrap loop rather than in the bootstrap orchestration itself.

## JAR Checksum Summary

| Date | Commit | MD5 | Size (bytes) | Status |
|---|---|---|---|---|
| 2026-01-30 | `31c01e9` | `42a99f5a612ad16f5b4575dc85f1c8dd` | 50,209,103 | Working (up to 500/1000 resamplings) |
| 2026-03-21 | `8e2a038` | `310f130bd7be273a33905c291aea7bf3` | 53,049,153 | `graphToJson` broken |
| 2026-03-23 | `00e200e` | `a3b57483751b8a235004e23dacdfb7a4` | 53,029,188 | `graphToJson` fixed, resampling hangs > 50 |
| 2026-03-27 | `bea3e8a` | `307b90e7db976c3e5566a3167c278868` | 50,764,069 | Current — resampling still hangs > 50 |

## Additional Notes

- No JVM heap arguments (`-Xmx`) were passed — default JVM heap is used. This configuration worked previously with much higher resampling counts.
- The hang is not a performance degradation — the process never completes regardless of wait time.
- Both algorithms (FGES and BOSS) exhibit the same behavior, suggesting the issue is in shared bootstrap infrastructure.
- The issue was reproduced independently on **macOS (Darwin 24.6.0, arm64)** and **Windows**, ruling out platform-specific causes.
