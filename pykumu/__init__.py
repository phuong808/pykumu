# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pykumu API for causal analysis.

This package provides a modular Python API for causal discovery
using the Tetrad library via JPype. Each module exposes thin
wrappers around Tetrad's Java classes:

- `pykumu.data` -- load and prepare data for causal search
- `pykumu.score` -- configure scoring functions (e.g., SEM BIC)
- `pykumu.bootstrapping` -- configure bootstrap resampling parameters
- `pykumu.knowledge` -- load domain knowledge constraints
- `pykumu.algorithm` -- run causal search algorithms (FGES, BOSS)
- `pykumu.graph` -- serialize and parse Tetrad graph objects, apply PNEF thresholds
- `pykumu.translate` -- convert between pandas and Tetrad data formats

## Usage

Call `pykumu.tetrad()` once before using any other module to start the
JVM and load the Tetrad JAR::

    import pykumu
    pykumu.tetrad("path/to/tetrad-current.jar")

    from pykumu import data, score, algorithm

## Vignettes

- [Issue Causal Analysis](/pykumu/vignettes/issue_causal_analysis.html) --
  end-to-end causal discovery workflow demonstrating data preparation,
  null-variable bootstrapped search, 1-PNEF threshold derivation,
  and final causal graph inspection.
"""

import os
import warnings

import jpype
import jpype.imports


def tetrad(jar_path, jvm_args=None):
    """Start the JVM and load the Tetrad JAR.

    Must be called once before using any pykumu module that accesses
    Tetrad Java classes (data, score, algorithm, etc.).

    :param jar_path: Path to the ``tetrad-current.jar`` file.
    :param jvm_args: Optional list of extra JVM arguments
        (e.g. ``["-Xmx8g"]`` to increase heap memory).
    :raises FileNotFoundError: If *jar_path* does not exist.
    """
    jar_path = os.path.abspath(jar_path)
    if not os.path.isfile(jar_path):
        raise FileNotFoundError(f"Tetrad JAR not found: {jar_path}")

    if jpype.isJVMStarted():
        warnings.warn(
            "JVM is already running. tetrad() has no effect after the "
            "first call — restart the Python process to change the JAR.",
            stacklevel=2,
        )
        return

    args = [jpype.getDefaultJVMPath()]
    if jvm_args:
        args.extend(jvm_args)

    jpype.startJVM(*args, classpath=[jar_path])


__all__ = [
    "tetrad",
    "algorithm",
    "bootstrapping",
    "data",
    "graph",
    "knowledge",
    "score",
    "translate",
]
