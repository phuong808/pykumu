import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp


"""
    Pykumu - https://github.com/sailuh/pykumu
    
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

"""
    Convert a Tetrad graph object to a JSON string.

    :param graph: Tetrad Java graph object (e.g., from algorithm.run_fges()['graph'])
    :returns: JSON string representation of the graph
    
"""
def get_json(graph):
    
    return str(gp.graphToJson(graph))
