"""Knowledge loading and constraint helpers."""

from __future__ import annotations


def load_knowledge(search, knowledge_file: str | None = None) -> None:
    """Load a Tetrad knowledge file if one is provided."""
    if knowledge_file:
        search.load_knowledge(knowledge_file)
