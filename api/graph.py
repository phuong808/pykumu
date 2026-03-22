"""Graph parsing and edge-threshold helpers."""

from __future__ import annotations

import json
import pandas as pd


def parse_graph(graph_json_str) -> dict[str, pd.DataFrame]:
    """Parse Tetrad JSON graph into nodes, edgeset, and edge_type_probabilities."""
    graph_data = json.loads(str(graph_json_str))

    nodes = pd.DataFrame({
        "node_name": [node["name"] for node in graph_data.get("nodes", [])]
    })

    edgeset_rows = []
    edge_type_prob_rows = []

    for edge in graph_data.get("edgesSet", []):
        node1_name = edge["node1"]["name"]
        node2_name = edge["node2"]["name"]
        endpoint1 = edge.get("endpoint1", "")
        endpoint2 = edge.get("endpoint2", "")
        bold = edge.get("bold", False)
        highlighted = edge.get("highlighted", False)
        properties = ";".join(edge.get("properties", []))
        probability = edge.get("probability", 1.0)

        edgeset_rows.append(
            {
                "node1_name": node1_name,
                "node2_name": node2_name,
                "endpoint1": endpoint1,
                "endpoint2": endpoint2,
                "bold": bold,
                "highlighted": highlighted,
                "properties": properties if properties else None,
                "probability": probability,
            }
        )

        for etp in edge.get("edgeTypeProbabilities", []):
            edge_type_prob_rows.append(
                {
                    "node1_name": node1_name,
                    "node2_name": node2_name,
                    "edge_type": etp.get("edgeType", ""),
                    "properties": ";".join(etp.get("properties", [])) or None,
                    "probability": etp.get("probability", 0.0),
                }
            )

    edgeset = (
        pd.DataFrame(edgeset_rows)
        if edgeset_rows
        else pd.DataFrame(
            columns=[
                "node1_name",
                "node2_name",
                "endpoint1",
                "endpoint2",
                "bold",
                "highlighted",
                "properties",
                "probability",
            ]
        )
    )

    edge_type_probabilities = (
        pd.DataFrame(edge_type_prob_rows)
        if edge_type_prob_rows
        else pd.DataFrame(
            columns=["node1_name", "node2_name", "edge_type", "properties", "probability"]
        )
    )

    return {
        "nodes": nodes,
        "edgeset": edgeset,
        "edge_type_probabilities": edge_type_probabilities,
    }


def filter_null_edges(edgeset: pd.DataFrame, null_prefix: str = "nv-") -> pd.DataFrame:
    """Subset edges where at least one endpoint is a null variable."""
    is_node1_nv = edgeset["node1_name"].str.contains(null_prefix, regex=False)
    is_node2_nv = edgeset["node2_name"].str.contains(null_prefix, regex=False)
    return edgeset[is_node1_nv | is_node2_nv].copy()


def add_no_edge_probability(edgeset: pd.DataFrame) -> pd.DataFrame:
    """Add no_edge = 1 - probability."""
    out = edgeset.copy()
    out["no_edge"] = 1 - out["probability"]
    return out


def pnef_threshold(edgeset: pd.DataFrame, quantile: float = 0.01) -> float:
    """Compute the PNEF threshold from no_edge probabilities."""
    with_no_edge = add_no_edge_probability(edgeset)
    return float(with_no_edge["no_edge"].quantile(quantile))


def apply_pnef(edgeset: pd.DataFrame, pnef_value: float) -> pd.DataFrame:
    """Filter edges using no_edge <= pnef_value."""
    with_no_edge = add_no_edge_probability(edgeset)
    return with_no_edge[with_no_edge["no_edge"] <= pnef_value].copy()
