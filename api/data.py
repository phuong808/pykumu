import api.translate as tr
import edu.cmu.tetrad.data as td
from edu.cmu.tetrad.util import Parameters


"""
    Pykumu - https://github.com/sailuh/pykumu
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

"""
    Convert a pandas DataFrame to Tetrad data format and initialize search state.

    :param df: pandas DataFrame with all columns as float (continuous)
    :returns: dict with 'data', 'params', 'knowledge' keys
    
"""
def load_continuous(df):
    
    data = tr.pandas_data_to_tetrad(df)
    params = Parameters()
    knowledge = td.Knowledge()
    return {"data": data, "params": params, "knowledge": knowledge}
