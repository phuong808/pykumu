"""Tetrad JVM initialization helpers."""

from __future__ import annotations

from pathlib import Path
import os

import jpype
import jpype.imports


def initialize_tetrad_jvm(repo_root: str | None = None, working_dir: str | None = None) -> Path:
    """Initialize JVM with tetrad-current.jar and return the resolved jar path."""
    _ = jpype.imports
    notebook_dir = Path(working_dir or os.getcwd())
    root = Path(repo_root) if repo_root else notebook_dir.parent

    jar_candidates = [
        notebook_dir / "resources" / "tetrad-current.jar",
        root / "resources" / "tetrad-current.jar",
        root / "pytetrad" / "resources" / "tetrad-current.jar",
    ]
    jar_path = next((p for p in jar_candidates if p.exists()), None)
    if jar_path is None:
        raise FileNotFoundError(
            "Could not find tetrad-current.jar. Checked: " + ", ".join(str(p) for p in jar_candidates)
        )

    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[str(jar_path)])

    return jar_path
