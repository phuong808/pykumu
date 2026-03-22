"""Translation helpers between pandas objects and Tetrad Java objects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from jpype import JClass


def pandas_data_to_tetrad(df: pd.DataFrame):
    """Convert a pandas DataFrame into a Tetrad BoxDataSet (continuous)."""
    data = df.copy()

    # Tetrad search wrappers in this repo use continuous scores/tests for this notebook.
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    values = data.to_numpy(dtype=float)
    nrows, ncols = values.shape

    ArrayList = JClass("java.util.ArrayList")
    ContinuousVariable = JClass("edu.cmu.tetrad.data.ContinuousVariable")
    DoubleDataBox = JClass("edu.cmu.tetrad.data.DoubleDataBox")
    BoxDataSet = JClass("edu.cmu.tetrad.data.BoxDataSet")

    variables = ArrayList()
    for name in data.columns:
        variables.add(ContinuousVariable(str(name)))

    box = DoubleDataBox(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            val = values[i, j]
            if np.isnan(val):
                val = float("nan")
            box.set(i, j, float(val))

    return BoxDataSet(box, variables)


