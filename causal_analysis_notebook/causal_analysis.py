# %% [markdown]
# # Causal Discovery Analysis
#
# **Reference:** Kumu R package, `issue_causal_analysis.Rmd`

# %% [markdown]
# ## Notebook Setup Instructions
#
# ### Environment Requirements
#
# - **OS:** macOS/Linux/Windows (tested in this repo on macOS).
# - **Python:** 3.10+ recommended (repo requirement is `>=3.7`; avoid very old Python).
# - **Java:** JDK 11+ required (JDK 17 works well). The notebook allocates up to 8 GB of Java heap — ensure your machine has at least 10 GB of available RAM.
# - **Jupyter:** VS Code Notebook or JupyterLab.
#
# ### Required Python Packages
#
# Install from the repository root (`pykumu/`) in a clean virtual environment:
#
# ```bash
# python3 -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# python -m pip install --upgrade pip
# pip install numpy pandas JPype1
# ```
#
# Install the local `pytetrad` package from the repo root:
#
# ```bash
# pip install -e .
# ```
#
# **Optional packages** (only needed if uncommenting the visualisation cells):
#
# ```bash
# pip install pyvis networkx
# ```
#
# ### Required Files (in `causal_analysis_notebook/`)
#
# - `null_variable_dt.csv` — input dataset (raw or preprocessed format both supported)
# - `mike_knowledge_box.txt` — domain-knowledge constraints for the causal search
#
# ### Working Directory and Kernel
#
# - Open notebook: `causal_analysis_notebook/causal_analysis.ipynb`.
# - The notebook uses **relative file paths**, so the working directory must resolve to `causal_analysis_notebook/`. In VS Code this is automatic when opening the file directly; in JupyterLab, launch from inside the folder.
# - Select the Python interpreter from the virtual environment where dependencies were installed.
# - If helper modules change, **restart the kernel** before re-running.
#
# ### Configuration Checklist (Configuration cell)
#
# - `ALGORITHM`: `"boss"` or `"fges"`
# - `DATA_PATH`: path to the input CSV (default: `null_variable_dt.csv`); change this to use a different dataset without modifying anything else
# - `KNOWLEDGE_FILE`: verify the file exists (default: `mike_knowledge_box.txt`)
# - `N_BOOTSTRAP`: lower for quick validation runs (e.g. `10`), increase for full analysis
#
# ### Recommended Run Order
#
# 1. Run **Configuration** and **Import Libraries**.
# 2. Run all **Feature Engineering** cells top-to-bottom.
# 3. Run **FGES Null Variable Search**.
# 4. Run **Deriving the 1 PNEF Threshold** (`pnef_1` must be defined before the next step).
# 5. Run **Non-Null Causal Search**.
# 6. Run **Applying 1PNEF Threshold**.
# 7. Run **Results** sections.
#
# ### Troubleshooting
#
# - **`AttributeError` on Tetrad methods:** restart the kernel and rerun from the top (JPype module reload issue).
# - **Java heap / out-of-memory errors:** reduce `N_BOOTSTRAP` and rerun.
# - **File not found:** confirm the working directory is `causal_analysis_notebook/` and that the required CSV and knowledge files are present.
# - **Import errors (`pytetrad`, `jpype`):** confirm the active interpreter is the virtual environment where the packages were installed.

# %% [markdown]
# ## Configuration
#
# Set algorithm parameters and file paths.

# %%
# Analysis Parameters
ALGORITHM = "boss"  # Options: "boss" or "fges"

# Input/output paths
DATA_PATH = "null_variable_dt.csv"  # Input CSV (raw or preprocessed format)

KNOWLEDGE_FILE = "mike_knowledge_box.txt"

# Algorithm-specific parameters
N_BOOTSTRAP = 80 if ALGORITHM == "boss" else 50

# %% [markdown]
# ## Import Libraries
#
# Load required modules for data processing and causal analysis.

# %%
# Increase Java memory allocation for large bootstrap analyses
import os
os.environ['JAVA_TOOL_OPTIONS'] = '-Xmx8g -Xms4g'  # 8GB max heap, 4GB initial

import sys

# Ensure the local pytetrad (which has get_json) takes precedence over any
# system-installed version. The repo root is one level above this notebook.
_repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pandas as pd
import numpy as np
import json
import time
import api.TetradSearch as ts

print(f"pytetrad loaded from: {ts.__file__}")


def parse_graph(graph_json_str):
    """
    Parse Tetrad JSON graph into nodes, edgeset, and edge_type_probabilities.

    Matches R function: parse_graph() from kumu.

    The edgeset table contains the ensemble edge for each node pair. Because we
    are performing multiple executions (bootstrap), the probabilities represent
    the ensemble of all edges formed on each execution.
    """
    graph_data = json.loads(str(graph_json_str))

    nodes = pd.DataFrame({
        'node_name': [node['name'] for node in graph_data.get('nodes', [])]
    })

    edgeset_rows = []
    edge_type_prob_rows = []

    for edge in graph_data.get('edgesSet', []):
        node1_name  = edge['node1']['name']
        node2_name  = edge['node2']['name']
        endpoint1   = edge.get('endpoint1', '')
        endpoint2   = edge.get('endpoint2', '')
        bold        = edge.get('bold', False)
        highlighted = edge.get('highlighted', False)
        properties  = ';'.join(edge.get('properties', []))
        probability = edge.get('probability', 1.0)

        edgeset_rows.append({
            'node1_name':  node1_name,
            'node2_name':  node2_name,
            'endpoint1':   endpoint1,
            'endpoint2':   endpoint2,
            'bold':        bold,
            'highlighted': highlighted,
            'properties':  properties if properties else None,
            'probability': probability
        })

        for etp in edge.get('edgeTypeProbabilities', []):
            edge_type_prob_rows.append({
                'node1_name': node1_name,
                'node2_name': node2_name,
                'edge_type':  etp.get('edgeType', ''),
                'properties': ';'.join(etp.get('properties', [])) or None,
                'probability': etp.get('probability', 0.0)
            })

    edgeset = pd.DataFrame(edgeset_rows) if edgeset_rows else pd.DataFrame(
        columns=['node1_name', 'node2_name', 'endpoint1', 'endpoint2',
                 'bold', 'highlighted', 'properties', 'probability'])

    edge_type_probabilities = pd.DataFrame(edge_type_prob_rows) if edge_type_prob_rows else pd.DataFrame(
        columns=['node1_name', 'node2_name', 'edge_type', 'properties', 'probability'])

    return {
        'nodes': nodes,
        'edgeset': edgeset,
        'edge_type_probabilities': edge_type_probabilities
    }

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ## Formatting Data Types
#
# In order to be loaded in Tetrad, some variables must be transformed from String to Integer due to data type limitations.

# %% [markdown]
# ### CVE Data Type
#
# We concatenate the last two digits of the year with the last four digits of the cve_id and convert into an integer. (E.g. 2006 and CVE ID XXX4339 becomes 06339).

# %%
raw_dt = pd.read_csv(DATA_PATH)
print(f"Loaded: {raw_dt.shape[0]} rows × {raw_dt.shape[1]} columns")

# CVE Data Type: encode cve_id if present; otherwise reconstruct from b_ indicator columns
if "cve_id" in raw_dt.columns:
    cve_as_str = raw_dt["cve_id"].astype(str)
    cve_id_col = pd.to_numeric(cve_as_str.str.slice(6, 8) + cve_as_str.str.slice(-4), errors="coerce")
else:
    b_cols_present = [c for c in raw_dt.columns if c.startswith("b_")]
    if b_cols_present:
        b_mat = raw_dt[b_cols_present].to_numpy()
        b_idx = b_mat.argmax(axis=1)
        has_signal = b_mat.sum(axis=1) > 0
        cve_id_col = [
            int(b_cols_present[b_idx[r]].replace("b_", "")) if has_signal[r] else r
            for r in range(len(raw_dt))
        ]
    else:
        cve_id_col = range(len(raw_dt))

# Activity features: derive from commit_interval if present, else from commit counts
if "commit_interval" in raw_dt.columns:
    commit_interval = raw_dt["commit_interval"].fillna("").astype(str)
    activity_0_col = np.where(commit_interval.eq(""), 1, 0)
    activity_2_col = np.where(commit_interval.ne(""), 1, 0)
elif "commit" in raw_dt.columns:
    activity_0_col = np.where(raw_dt["commit"] <= 0, 1, 0)
    activity_2_col = np.where(raw_dt["commit"] > 0, 1, 0)
else:
    activity_0_col = 0
    activity_2_col = 0

# Assign all new/updated columns at once to avoid fragmentation warnings
raw_dt = raw_dt.assign(cve_id=cve_id_col, activity_0=activity_0_col, activity_2=activity_2_col)
print("Completed: CVE Data Type")

# %% [markdown]
# ### Convert "start" to Unix Timestamp
#
# To use start in causal analysis, we convert it to a unix timestamp.

# %%
# Convert "start" to Unix Timestamp
# If already numeric (unix timestamp), leave it as-is; otherwise parse from string/datetime
if "start_datetime" in raw_dt.columns:
    raw_dt["start"] = pd.to_datetime(raw_dt["start_datetime"], errors="coerce", utc=True)
    raw_dt["start"] = raw_dt["start"].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
elif "start" in raw_dt.columns:
    if pd.api.types.is_numeric_dtype(raw_dt["start"]):
        pass  # already a Unix timestamp
    else:
        raw_dt["start"] = pd.to_datetime(raw_dt["start"], errors="coerce", utc=True)
        raw_dt["start"] = raw_dt["start"].map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
else:
    raise ValueError("Expected either 'start_datetime' or 'start' column in data")

print("Completed: Convert 'start' to Unix Timestamp")

# %% [markdown]
# ## Feature Renaming

# %%
# Feature Renaming: shorten long column names; skip columns not present
rename_map = {
    "start_datetime": "start",
    "missing_links": "mis_link",
    "radio_silence": "silence",
    "code_only_devs": "code_dev",
    "code_files": "file",
    "ml_only_devs": "mail_dev",
    "ml_threads": "thread",
    "n_commits": "commit"
}
raw_dt = raw_dt.rename(columns={k: v for k, v in rename_map.items() if k in raw_dt.columns})

# Collect all missing required columns and assign them at once
required_cols = [
    "cve_id", "activity_0", "activity_2", "start",
    "org_silo", "mis_link", "silence", "code_dev", "file",
    "mail_dev", "thread", "commit", "churn"
]
missing_cols = {col: 0 for col in required_cols if col not in raw_dt.columns}
if missing_cols:
    raw_dt = raw_dt.assign(**missing_cols)

dt = raw_dt[required_cols].copy()
print(f"After Feature Renaming: {dt.shape[0]} rows × {dt.shape[1]} columns")

# %% [markdown]
# ## Missing Data Handling
#
# We decided to remove rows from the dataset for which the mailing list data source is missing (i.e. 2000-2001).

# %%
# Convert start to datetime for year-based filtering, handling both
# numeric Unix timestamps and string datetime formats
if pd.api.types.is_numeric_dtype(dt["start"]):
    start_dt_series = pd.to_datetime(dt["start"], unit='s', utc=True, errors='coerce')
else:
    start_dt_series = pd.to_datetime(dt["start"], errors="coerce", utc=True)

mask = (start_dt_series.dt.year < 2000) | (start_dt_series.dt.year > 2001)
dt = dt[mask].copy()
start_dt_series = start_dt_series[mask]

dt["start"] = start_dt_series.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)
dt = dt.fillna(0)
print(f"After Missing Data Transformations: {dt.shape[0]} rows × {dt.shape[1]} columns")

# %% [markdown]
# ## 1-Time Lag Features

# %%
lag_feature_cols = [
    "org_silo", "mis_link", "silence", "code_dev", "file",
    "mail_dev", "thread", "commit", "churn"
]

dt = dt.sort_values(["cve_id", "start"]).reset_index(drop=True)

def add_time_lag(cve_table: pd.DataFrame) -> pd.DataFrame:
    if len(cve_table) < 2:
        lag_cols = pd.DataFrame(
            {f"{col}2": np.nan for col in lag_feature_cols},
            index=cve_table.index
        )
        return pd.concat([cve_table, lag_cols], axis=1)

    lag_cols = pd.DataFrame(
        {f"{col}2": cve_table[col].shift(-1).values for col in lag_feature_cols},
        index=cve_table.index
    )
    result = pd.concat([cve_table, lag_cols], axis=1)
    return result.iloc[:-1].copy()

lag_parts = [add_time_lag(group) for _, group in dt.groupby("cve_id", sort=False)]
lag_dt = pd.concat(lag_parts, ignore_index=True)
print(f"After Appending Next Time Period Variables: {lag_dt.shape[0]} rows × {lag_dt.shape[1]} columns")

# %% [markdown]
# ## Remove Short CVEs
#
# We deleted CVEs (their associated rows) with 7 or fewer time periods.

# %%
cve_counts = lag_dt.groupby("cve_id").size()
short_cve_ids = cve_counts[cve_counts <= 7].index
lag_dt = lag_dt[~lag_dt["cve_id"].isin(short_cve_ids)].copy()
print(f"After Removing Short CVEs: {lag_dt.shape[0]} rows × {lag_dt.shape[1]} columns")

# %% [markdown]
# ## Addressing Determinism and High Intercorrelation Among Features
#
# Due to high correlation, we perform 6 feature deletions (activity_0, activity_2, org_silo, org_silo2):

# %%
selected_cols = [
    "cve_id", "start",
    "mis_link", "silence", "code_dev", "file",
    "mail_dev", "thread", "commit", "churn",
    "mis_link2", "silence2", "code_dev2", "file2",
    "mail_dev2", "thread2", "commit2", "churn2"
]
lag_dt = lag_dt[selected_cols].copy()
print(f"After Correlation/Determinism Pruning: {lag_dt.shape[0]} rows × {lag_dt.shape[1]} columns")

# %% [markdown]
# ## Binarized CVE Indicators
#
# To represent the CVE Ids, we utilize indicator features. For every CVE ID, a new column is added to the table which can take values 0 or 1. The value is 1 if the row is associated to that CVE ID, or 0 otherwise.
#
# We can then remove the `cve_id` column, as the binary features represent the same information, and add the remaining columns to the analysis table:

# %%
cve_binarized = pd.get_dummies(
    lag_dt["cve_id"].astype("Int64").astype(str),
    prefix="b",
    dtype=int
)

binarized_lag_dt = pd.concat([
    lag_dt.drop(columns=["cve_id"]).reset_index(drop=True),
    cve_binarized.reset_index(drop=True)
], axis=1)
print(f"After Binarize CVE ID: {binarized_lag_dt.shape[0]} rows × {binarized_lag_dt.shape[1]} columns")

# %% [markdown]
# ## Add Null Features
#
# ## Keep only 5 null indicator features
#
# Introducing a null feature for all variables and features leads to too many features being introduced for causal search, causing heap memory errors in Tetrad. We preserve only a few of the nv binary indicator variables, as they lead to variable explosion and their pattern is easy to randomize. Position 138 includes all variables as null variables, plus five binary indicators as null variables. We consider this loss of null binary indicator features reasonable, as the randomization of a few blocks of values 1 or 0 will generally be equivalent. This in turn, allow us to perform more causal search runs, which we deem a fair trade-off.

# %%
rng = np.random.default_rng(32)
null_df = pd.DataFrame(
    {col: rng.permutation(binarized_lag_dt[col].to_numpy()) for col in binarized_lag_dt.columns}
)
null_df.columns = [f"nv-{col}" for col in null_df.columns]

null_non_indicator_cols = [c for c in null_df.columns if not c.startswith("nv-b_")]
null_indicator_cols = [c for c in null_df.columns if c.startswith("nv-b_")]
null_keep_cols = null_non_indicator_cols + null_indicator_cols[:5]
null_df = null_df[null_keep_cols]

processed_dt = pd.concat([binarized_lag_dt, null_df], axis=1)

# Save outputs to processed_data/ — input DATA_PATH is never modified
os.makedirs("processed_data", exist_ok=True)
processed_dt.to_csv("processed_data/null_variable_processed_dt.csv", index=False)
binarized_lag_dt.to_csv("processed_data/binarized_variable_dt.csv", index=False)

print(f"✓ Saved null-variable dataset : processed_data/null_variable_processed_dt.csv ({processed_dt.shape[0]} × {processed_dt.shape[1]})")
print(f"✓ Saved non-null dataset      : processed_data/binarized_variable_dt.csv ({binarized_lag_dt.shape[0]} × {binarized_lag_dt.shape[1]})")

# Use processed data in-memory for the rest of the notebook
data = processed_dt.copy()
data = data.astype({col: "float64" for col in data.columns})
print(f"Dataset: {data.shape[0]} rows × {data.shape[1]} columns")

# %% [markdown]
# ## Variable Analysis
#
# Identify binary CVE indicators, continuous metrics, and null variables.

# %%
b_cols      = [col for col in data.columns if col.startswith('b_')]
nv_cols     = [col for col in data.columns if col.startswith('nv-')]
metric_cols = [col for col in data.columns if not col.startswith('b_') and not col.startswith('nv-')]

print(f"Binary indicators: {len(b_cols)}")
print(f"Continuous metrics: {len(metric_cols)}")
print(f"Null variables: {len(nv_cols)}")

# %% [markdown]
# # FGES Null Variable Search
#
# Executes causal discovery with bootstrapping over the null-variable dataset.

# %%
# Matches R: data_io + algorithm_boss/fges + score_sem_bic + bootstrapping + tetrad()
# Note: No domain knowledge constraints in null variable search (see R notebook)
search = ts.TetradSearch(data)
search.use_sem_bic(penalty_discount=2, sem_bic_rule=1,
                   structurePrior=0, singularity_lambda=0.0)

if ALGORITHM == "boss":
    search.set_bootstrapping(numberResampling=N_BOOTSTRAP, percent_resample_size=100,
                             seed=32, add_original=True, with_replacement=True,
                             resampling_ensemble=1)
    search.set_verbose(verbose=False)
else:
    search.set_bootstrapping(numberResampling=N_BOOTSTRAP, percent_resample_size=90,
                             seed=32, add_original=True, with_replacement=True,
                             resampling_ensemble=1)
    search.set_verbose(verbose=False)

import jpype
_System = jpype.JClass("java.lang.System")
_orig_out = _System.out
_System.setOut(jpype.JClass("java.io.PrintStream")(
    jpype.JClass("java.io.ByteArrayOutputStream")()
))

start_time = time.time()
try:
    if ALGORITHM == "boss":
        search.run_boss(num_starts=1, use_bes=False, time_lag=0,
                        use_data_order=True, output_cpdag=True)
    else:
        search.run_fges(max_degree=1000, faithfulness_assumed=True,
                        symmetric_first_step=True, parallelized=False)
finally:
    _System.setOut(_orig_out)

elapsed = time.time() - start_time
graph   = search.get_java()
print(f"Discovered: {graph.getNumNodes()} nodes, {graph.getNumEdges()} edges")
print(f"Elapsed: {elapsed:.1f}s")

null_graph_json = str(search.get_json())

# %%
import glob
import shutil

os.makedirs("logs", exist_ok=True)
log_files = glob.glob("*.log")
for f in log_files:
    shutil.move(f, os.path.join("logs", os.path.basename(f)))
if log_files:
    print(f"Moved {len(log_files)} log file(s) to logs/")

# %% [markdown]
# ---
#
# # Deriving the 1 PNEF Threshold
#
# In our causal search above, we introduced null features over multiple bootstrap runs to observe how often our causal search forms random edges (i.e. between our features and null features). We will use this information to derive a threshold, **1PNEF** (1st Percentile NoEdge Frequency), we can use in our final causal search.
#
# ## Graph Examination
#
# We parse the Tetrad JSON graph output into tabular format: nodes, edgeset, and edge type probabilities.
#
# The **edgeset** table contains the ensemble edge for each node pair. Because we performed multiple bootstrap runs, the probabilities represent the ensemble of all edges formed on each execution.
#
# The **edge_type_probabilities** table shows the counts of each type of edge formed on each subgraph across all bootstrap runs.

# %%
null_graph = parse_graph(null_graph_json)

print(f"Nodes: {len(null_graph['nodes'])}")
print(f"\nFirst 5 nodes:")
print(null_graph['nodes'].head())
print(f"\nEdgeset: {len(null_graph['edgeset'])} edges")
print(null_graph['edgeset'].head())
print(f"\nEdge type probabilities: {len(null_graph['edge_type_probabilities'])} entries")
print(null_graph['edge_type_probabilities'].head())

# %% [markdown]
# ## Deriving 1 PNEF
#
# Our interest is to derive a threshold for the final causal search, using the information from this bootstrapped null feature causal search. By definition, edges formed between actual variables and random (null) features represent random edges.
#
# We:
# 1. Subset the edgeset to contain only edges where at least one node is a null variable (nv-*)
# 2. Derive a `no_edge` probability by subtracting the probability from 1
# 3. Identify the 1st percentile value of the no_edge probability → the **1PNEF threshold**
#
# This threshold tells us: given entirely random variables, causal links were formed between them up to X% of the time. In our final search, we only keep causal links that formed **more** than X% of the time.

# %%
nv_edges = null_graph['edgeset'].copy()
is_node1_nv = nv_edges['node1_name'].str.contains('nv-', regex=False)
is_node2_nv = nv_edges['node2_name'].str.contains('nv-', regex=False)
nv_edges = nv_edges[is_node1_nv | is_node2_nv]
nv_edges.head()

nv_edges['no_edge'] = 1 - nv_edges['probability']

pnef_1 = float(nv_edges['no_edge'].quantile(0.01))
pnef_1

# %% [markdown]
# ---
#
# # Non-Null Causal Search
#
# With the threshold defined, we now proceed to the final causal search, which **does not include null features**. In this non-null feature causal search, we also specify domain knowledge to prohibit causal links that don't make sense temporally (e.g. features at 1-time-lag cannot cause features in the present).
#
# ## Domain Knowledge Causal Search without Null Variables
#
# Remove null variable columns (nv-*) from the dataset, keeping only the original features and binary CVE indicators.

# %%
nv_cols       = [col for col in data.columns if col.startswith('nv-')]
non_null_data = data.drop(columns=nv_cols)
non_null_data.to_csv("processed_data/binarized_variable_dt.csv", index=False)
print(f"Non-null dataset: {non_null_data.shape[0]} rows × {non_null_data.shape[1]} columns")
print(f"Saved to: processed_data/binarized_variable_dt.csv")

# %% [markdown]
# ## Causal Search
#
# Run the causal search on the non-null dataset with domain knowledge constraints. This search uses the same algorithm and bootstrap settings, but on the dataset **without** null features and **with** temporal knowledge constraints.

# %%
# Matches R: data_io + knowledge_flags + algorithm_boss/fges + score_sem_bic + bootstrapping + tetrad()
# Domain knowledge prohibits lag-2 features from causing lag-1 features (temporal ordering)
domain_search = ts.TetradSearch(non_null_data)
domain_search.use_sem_bic(penalty_discount=2, sem_bic_rule=1,
                           structurePrior=0, singularity_lambda=0.0)

if ALGORITHM == "boss":
    domain_search.set_bootstrapping(numberResampling=N_BOOTSTRAP, percent_resample_size=100,
                                     seed=32, add_original=True, with_replacement=True,
                                     resampling_ensemble=1)
    domain_search.set_verbose(verbose=False)
    if os.path.exists(KNOWLEDGE_FILE):
        domain_search.load_knowledge(KNOWLEDGE_FILE)
        print(f"Knowledge loaded from: {KNOWLEDGE_FILE}")
else:
    domain_search.set_bootstrapping(numberResampling=N_BOOTSTRAP, percent_resample_size=90,
                                     seed=32, add_original=True, with_replacement=True,
                                     resampling_ensemble=1)
    domain_search.set_verbose(verbose=False)
    if os.path.exists(KNOWLEDGE_FILE):
        domain_search.load_knowledge(KNOWLEDGE_FILE)
        print(f"Knowledge loaded from: {KNOWLEDGE_FILE}")

_orig_out = _System.out
_System.setOut(jpype.JClass("java.io.PrintStream")(
    jpype.JClass("java.io.ByteArrayOutputStream")()
))

start_time = time.time()
try:
    if ALGORITHM == "boss":
        domain_search.run_boss(num_starts=1, use_bes=False, time_lag=0,
                               use_data_order=True, output_cpdag=True)
    else:
        domain_search.run_fges(max_degree=1000, faithfulness_assumed=True,
                               symmetric_first_step=True, parallelized=False)
finally:
    _System.setOut(_orig_out)

elapsed          = time.time() - start_time
domain_graph_obj = domain_search.get_java()
print(f"Discovered: {domain_graph_obj.getNumNodes()} nodes, {domain_graph_obj.getNumEdges()} edges")
print(f"Elapsed: {elapsed:.1f}s")

domain_graph_json = str(domain_search.get_json())

# %%
import glob
import shutil

os.makedirs("logs", exist_ok=True)
log_files = glob.glob("*.log")
for f in log_files:
    shutil.move(f, os.path.join("logs", os.path.basename(f)))
if log_files:
    print(f"Moved {len(log_files)} log file(s) to logs/")

# %% [markdown]
# ## Graph Examination
#
# Parse the domain knowledge causal search JSON output into nodes, edgeset, and edge type probabilities.

# %%
# Parse the domain search JSON output
domain_graph = parse_graph(domain_graph_json)

print(f"Domain search nodes: {len(domain_graph['nodes'])}")
print(f"Domain search edges: {len(domain_graph['edgeset'])}")
print(f"\nEdgeset sample:")
domain_graph['edgeset'].head()

# %% [markdown]
# ---
#
# # Applying 1PNEF Threshold
#
# ## Applying 1PNEF Threshold
#
# Edges which may have been formed at random are filtered here. We apply the 1PNEF threshold derived from the null variable search to the domain knowledge search results. Only edges whose `no_edge` probability is less than or equal to the 1PNEF threshold are kept.

# %%
# Applying 1PNEF Threshold (R-equivalent explicit steps)
edges = domain_graph['edgeset'].copy()
edges['no_edge'] = 1 - edges['probability']
edges_1pnef = edges[edges['no_edge'] <= pnef_1].copy()

print(f"\nFiltered edges (1PNEF trimmed): {len(edges_1pnef)}")
edges_1pnef.head(20)

# %%
# Save the 1PNEF-filtered edges
output_dir = "boss_domain_results" if ALGORITHM == "boss" else "fges_domain_results"
os.makedirs(output_dir, exist_ok=True)
edges_1pnef.to_csv(f"{output_dir}/edges_1pnef.csv", index=False)
print(f"✓ Saved filtered edges to {output_dir}/edges_1pnef.csv")

# %% [markdown]
# ---
#
# # Results
#
# With the final causal graph trimmed, we can now inspect it to draw conclusions. Causal graphs may form cycles and have undirected edges.
#
# ## Full Causal Graph 1-PNEF Trimmed
#
# Interactive visualization of the full causal graph after applying the 1PNEF threshold.
#
# Edge colors:
# - **Black**: Directed edges (causal relationship)
# - **Red**: Undirected edges (TAIL-TAIL, association without determined direction)

# %%
# try:
#     from pyvis.network import Network
#     _pyvis_available = True
# except ImportError:
#     print("⚠️  pyvis not installed. Install with: pip install pyvis")
#     _pyvis_available = False
#
# # Color scheme matching R colorBlindness::Blue2DarkRed12Steps
# _COLORBLIND_PALETTE = {
#     'b_':       '#2166AC',  # Blue — binary indicators
#     'mis_link': '#67A9CF',  # Light blue
#     'silence':  '#D1E5F0',  # Pale blue
#     'code_dev': '#FDDBC7',  # Pale orange
#     'churn':    '#F4A582',  # Light red
#     'commit':   '#D6604D',  # Red
#     'default':  '#67A9CF',  # Default: light blue
# }
#
# def _node_color(name):
#     for key, color in _COLORBLIND_PALETTE.items():
#         if key != 'default' and key in name:
#             return color
#     return _COLORBLIND_PALETTE['default']
#
# # Prepare edges for visualization
# viz_edges = edges_1pnef.copy()
# viz_edges['color'] = 'black'
# viz_edges.loc[(viz_edges['endpoint1'] == 'TAIL') & (viz_edges['endpoint2'] == 'TAIL'), 'color'] = 'red'
# viz_edges = viz_edges.rename(columns={'node1_name': 'from', 'node2_name': 'to'})
# viz_edges['weight'] = viz_edges['probability']
# viz_edges['label']  = viz_edges['probability'].round(3).astype(str)
#
# if _pyvis_available and len(viz_edges) > 0:
#     net = Network(height="700px", width="100%", directed=True,
#                   notebook=True, cdn_resources='in_line')
#
#     all_node_names = set(viz_edges['from'].tolist() + viz_edges['to'].tolist())
#     all_node_names.update(domain_graph['nodes']['node_name'].tolist())
#     for n in all_node_names:
#         net.add_node(n, label=n, color=_node_color(n), title=n, size=20)
#
#     for _, row in viz_edges.iterrows():
#         arrows = 'to' if row.get('endpoint2') == 'ARROW' else (
#                  'from' if row.get('endpoint1') == 'ARROW' else '')
#         net.add_edge(row['from'], row['to'],
#                      color=row['color'],
#                      value=float(row['weight']),
#                      title=f"p={row['weight']}",
#                      label=row['label'],
#                      arrows=arrows)
#
#     net.set_options("""
#     {
#       "physics": {
#         "forceAtlas2Based": {
#           "gravitationalConstant": -50,
#           "centralGravity": 0.01,
#           "springLength": 200,
#           "springConstant": 0.08
#         },
#         "solver": "forceAtlas2Based",
#         "stabilization": {"iterations": 150}
#       },
#       "interaction": {"navigationButtons": true, "keyboard": true, "hover": true}
#     }
#     """)
#
#     output_html = f"{NON_NULL_OUTPUT_DIR}/causal_graph_full.html"
#     os.makedirs(NON_NULL_OUTPUT_DIR, exist_ok=True)
#     net.save_graph(output_html)
#     print(f"Graph saved to: {output_html}")
#     net.show(output_html)
# else:
#     print(f"\nFull causal graph: {len(viz_edges)} edges")
#     print(viz_edges[['from', 'to', 'probability', 'color']].to_string(index=False))

# %% [markdown]
# <!-- ## Sub-Graphs of Effort Variables and Parents
#
# Focus on key effort variables and their neighboring causal structure in a smaller sub-graph. -->

# %%
# if _pyvis_available and len(viz_edges) > 0:
#     # Include edges between nodes of interest AND edges from parents into nodes of interest
#     sub_mask = (
#         (viz_edges['from'].isin(NODES_OF_INTEREST) & viz_edges['to'].isin(NODES_OF_INTEREST)) |
#         viz_edges['to'].isin(NODES_OF_INTEREST)
#     )
#     sub_edges = viz_edges[sub_mask].copy()
#
#     if len(sub_edges) == 0:
#         print("No edges found between specified nodes of interest.")
#     else:
#         sub_node_names = set(sub_edges['from'].tolist() + sub_edges['to'].tolist())
#         print(f"Sub-graph: {len(sub_node_names)} nodes, {len(sub_edges)} edges")
#
#         sub_net = Network(height="700px", width="100%", directed=True,
#                           notebook=True, cdn_resources='in_line')
#
#         for n in sub_node_names:
#             sub_net.add_node(n, label=n, color=_node_color(n), title=n, size=20)
#
#         for _, row in sub_edges.iterrows():
#             arrows = 'to' if row.get('endpoint2') == 'ARROW' else (
#                      'from' if row.get('endpoint1') == 'ARROW' else '')
#             sub_net.add_edge(row['from'], row['to'],
#                              color=row['color'],
#                              value=float(row['weight']),
#                              title=f"p={row['weight']}",
#                              label=row['label'],
#                              arrows=arrows)
#
#         sub_net.set_options("""
#         {
#           "physics": {
#             "forceAtlas2Based": {
#               "gravitationalConstant": -50,
#               "centralGravity": 0.01,
#               "springLength": 200,
#               "springConstant": 0.08
#             },
#             "solver": "forceAtlas2Based",
#             "stabilization": {"iterations": 150}
#           },
#           "interaction": {"navigationButtons": true, "keyboard": true, "hover": true}
#         }
#         """)
#
#         sub_output_html = f"{NON_NULL_OUTPUT_DIR}/causal_graph_subgraph.html"
#         os.makedirs(NON_NULL_OUTPUT_DIR, exist_ok=True)
#         sub_net.save_graph(sub_output_html)
#         print(f"Sub-graph saved to: {sub_output_html}")
#         sub_net.show(sub_output_html)
# else:
#     print("No edges to display in sub-graph.")

# %% [markdown]
# <!-- ## Cycle Detection
#
# Check if the causal graph contains any cycles. Cycles indicate feedback loops in the causal structure. -->

# %%
# try:
#     import networkx as nx
#     _nx_available = True
# except ImportError:
#     print("⚠️  networkx not installed. Install with: pip install networkx")
#     _nx_available = False
#
# if _nx_available and len(viz_edges) > 0:
#     # Build directed graph from directed (non-TAIL-TAIL) edges only
#     G = nx.DiGraph()
#     directed_edges = viz_edges[viz_edges['color'] == 'black']
#     for _, row in directed_edges.iterrows():
#         G.add_edge(row['from'], row['to'])
#
#     cycles = list(nx.simple_cycles(G))
#
#     if not cycles:
#         print("No cycles detected in the graph.")
#     else:
#         print(f"Found {len(cycles)} cycle(s):")
#         for i, cycle in enumerate(cycles, 1):
#             cycle_str = " → ".join(cycle) + f" → {cycle[0]}"
#             print(f"  Cycle {i} (length {len(cycle)}): {cycle_str}")
# else:
#     print("Cycle detection skipped (networkx unavailable or no edges).")
