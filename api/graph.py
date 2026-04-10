# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Graph serialization and parsing utilities.

This module provides functions to convert Tetrad graph objects
into serializable formats (e.g., JSON) and parse Tetrad JSON graph
files into tabular DataFrames.
"""

import json

import pandas as pd
def get_json(graph):
    """Convert a Tetrad graph object to a JSON string.

    :param graph: Tetrad Java graph object (e.g., from algorithm.run_fges()['graph'])
    :returns: JSON string representation of the graph
    """
    import edu.cmu.tetrad.graph.GraphSaveLoadUtils as gp

    return str(gp.graphToJson(graph))


def convert_to_tetrad_gui_format(graph_json_str):
    """Convert a PyKumu/newer Tetrad JSON graph string to the format Tetrad GUI can load.

    The newer Tetrad library serializes nodeType as a string (e.g., "MEASURED"),
    while Tetrad GUI and causal-cmd expect it as an object (e.g., {"ordinal": 0}).
    This function converts between the two formats.

    :param graph_json_str: JSON string or file path to a graph JSON
    :returns: JSON string in Tetrad GUI-compatible format
    """
    NODE_TYPE_MAP = {
        "MEASURED": {"ordinal": 0},
        "LATENT": {"ordinal": 1},
        "ERROR": {"ordinal": 2},
    }

    if graph_json_str.endswith('.json'):
        with open(graph_json_str, 'r') as f:
            graph = json.load(f)
    else:
        graph = json.loads(graph_json_str)

    # Convert nodeType in all nodes
    def convert_node(node):
        if isinstance(node.get("nodeType"), str):
            node["nodeType"] = NODE_TYPE_MAP.get(node["nodeType"], {"ordinal": 0})
        # Remove extra fields Tetrad GUI doesn't expect
        node.pop("rank", None)
        node.pop("selectionBias", None)
        return node

    for node in graph.get("nodes", []):
        convert_node(node)

    # Convert nodes inside namesHash
    for name, node in graph.get("namesHash", {}).items():
        convert_node(node)

    # Convert nodes inside edges
    for edge in graph.get("edgesSet", []):
        convert_node(edge.get("node1", {}))
        convert_node(edge.get("node2", {}))

    # Convert nodes inside edgeLists
    for name, edges in graph.get("edgeLists", {}).items():
        for edge in edges:
            convert_node(edge.get("node1", {}))
            convert_node(edge.get("node2", {}))

    # Remove extra top-level keys Tetrad GUI doesn't expect
    for key in ["ancestorCache", "potentiallyDirectedPathCache", "ancillaryGraphs", "parentsHash"]:
        graph.pop(key, None)

    return json.dumps(graph, indent=2)


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
