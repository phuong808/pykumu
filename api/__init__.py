# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pykumu API for causal analysis.

This package provides a modular Python API for causal discovery
using the Tetrad library via JPype. Each module exposes thin
wrappers around Tetrad's Java classes:

- `api.data` -- load and prepare data for causal search
- `api.score` -- configure scoring functions (e.g., SEM BIC)
- `api.bootstrapping` -- configure bootstrap resampling parameters
- `api.knowledge` -- load domain knowledge constraints
- `api.algorithm` -- run causal search algorithms (FGES, BOSS)
- `api.graph` -- serialize and parse Tetrad graph objects, apply PNEF thresholds
- `api.translate` -- convert between pandas and Tetrad data formats

## Vignettes

- [Issue Causal Analysis](/pykumu/vignettes/issue_causal_analysis.html) --
  end-to-end causal discovery workflow demonstrating data preparation,
  null-variable bootstrapped search, 1-PNEF threshold derivation,
  and final causal graph inspection.
"""
