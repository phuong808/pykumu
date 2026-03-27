# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Graph serialization and parsing utilities.

This module provides functions to convert Tetrad graph objects
into serializable formats (e.g., JSON), parse Tetrad JSON graph
files into tabular DataFrames, and apply edge-threshold filters.
"""

import json

import pandas as pd
try:
    import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp
except ImportError:
    pass


def get_json(graph):
    """Convert a Tetrad graph object to a JSON string.

    :param graph: Tetrad Java graph object (e.g., from algorithm.run_fges()['graph'])
    :returns: JSON string representation of the graph
    """
    return str(gp.graphToJson(graph))


def parse_graph(graph_filepath):
    """Parse a Tetrad JSON graph file into nodes, edgeset, and edge_type_probabilities DataFrames.

    :param graph_filepath: File path to a Tetrad JSON graph file
    :returns: dict with 'nodes', 'edgeset', and 'edge_type_probabilities' DataFrames
    """
    with open(graph_filepath, 'r') as f:
        graph_json = json.load(f)

    nodes = pd.DataFrame({"node_name": [n["name"] for n in graph_json["nodes"]]})

    edgeset_rows = []
    etp_rows = []

    for edge in graph_json.get("edgesSet", []):
        node1_name = edge["node1"]["name"]
        node2_name = edge["node2"]["name"]
        endpoint1 = edge.get("endpoint1")
        endpoint2 = edge.get("endpoint2")
        bold = edge.get("bold")
        highlighted = edge.get("highlighted")
        properties = ";".join(edge.get("properties", [])) or None
        probability = edge.get("probability")

        edgeset_rows.append({
            "node1_name": node1_name, "node2_name": node2_name,
            "endpoint1": endpoint1, "endpoint2": endpoint2,
            "bold": bold, "highlighted": highlighted,
            "properties": properties, "probability": probability
        })

        for etp in edge.get("edgeTypeProbabilities", []):
            etp_props = ";".join(etp.get("properties", [])) or None
            etp_rows.append({
                "node1_name": node1_name, "node2_name": node2_name,
                "edge_type": etp.get("edgeType"),
                "properties": etp_props,
                "probability": etp.get("probability")
            })

    edgeset = pd.DataFrame(edgeset_rows)
    edge_type_probabilities = pd.DataFrame(etp_rows)

    return {"nodes": nodes, "edgeset": edgeset, "edge_type_probabilities": edge_type_probabilities}


def filter_null_edges(edgeset, null_prefix="nv-"):
    """Subset edges where at least one endpoint is a null variable.

    :param edgeset: DataFrame of edges (from parse_graph()['edgeset'])
    :param null_prefix: Prefix identifying null variable names
    :returns: DataFrame containing only edges with at least one null variable endpoint
    """
    is_node1_nv = edgeset["node1_name"].str.contains(null_prefix, regex=False)
    is_node2_nv = edgeset["node2_name"].str.contains(null_prefix, regex=False)
    return edgeset[is_node1_nv | is_node2_nv].copy()


def add_no_edge_probability(edgeset):
    """Add a no_edge column computed as 1 - probability.

    :param edgeset: DataFrame of edges with a 'probability' column
    :returns: DataFrame with an additional 'no_edge' column
    """
    out = edgeset.copy()
    out["no_edge"] = 1 - out["probability"]
    return out


def pnef_threshold(edgeset, quantile=0.01):
    """Compute the 1-PNEF threshold from null-variable edge no_edge probabilities.

    The 1st percentile NoEdge Frequency (1PNEF) threshold is derived from
    the no_edge probabilities of null-variable edges. Edges in the final
    causal graph with no_edge probability above this threshold may have
    formed by random chance and should be filtered out.

    :param edgeset: DataFrame of null-variable edges with a 'probability' column
    :param quantile: Quantile to use for the threshold (default 0.01 = 1st percentile)
    :returns: float threshold value
    """
    with_no_edge = add_no_edge_probability(edgeset)
    return float(with_no_edge["no_edge"].quantile(quantile))


def apply_pnef(edgeset, pnef_value):
    """Filter edges using the 1-PNEF threshold.

    Retains only edges whose no_edge probability is at or below the
    given PNEF threshold, removing edges that may have formed by random chance.

    :param edgeset: DataFrame of edges with a 'probability' column
    :param pnef_value: PNEF threshold value (from pnef_threshold())
    :returns: DataFrame of edges passing the threshold filter
    """
    with_no_edge = add_no_edge_probability(edgeset)
    return with_no_edge[with_no_edge["no_edge"] <= pnef_value].copy()
