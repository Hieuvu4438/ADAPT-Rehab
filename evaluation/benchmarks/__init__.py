"""
Benchmark dataset loaders for ADAPT-Rehab evaluation.

This package contains the canonical, importable loaders for the
rehabilitation-movement datasets used in the paper:

- :class:`UI_PRMDLoader` (``uiprmd.py``) — UI-PRMD (Univ. of Idaho)
- :class:`KimoreLoader` (``kimore.py``) — KIMORE (clinical scores)

Each loader exposes a uniform API:

    loader = UI_PRMDLoader(data_dir=...)     # or KimoreLoader(data_path=...)
    loader.is_available()                     # bool — dataset is on disk
    loader.load()                             # -> dataset in memory
    loader.summary()                          # -> dict of all metrics
    loader.iter_samples()                     # -> Iterator over samples

The legacy CLI scripts in ``scripts/eval_uiprmd.py`` and
``scripts/eval_kimore.py`` are thin wrappers that import from this
package and call its methods.
"""

from .uiprmd import UI_PRMDLoader
from .kimore import KimoreLoader

__all__ = [
    "UI_PRMDLoader",
    "KimoreLoader",
]
