"""
Lucas Sancéré 2025
"""

from __future__ import annotations
from typing import Any
import numpy as np


def _is_numeric(value: Any) -> bool:  # noqa: D401  – keep the underscores
    """Return *True* iff *value* is numeric or a sequence of numeric scalars.

    Parameters
    ----------
    value : Any
        Candidate value to test. Allowed containers are ``list``, ``tuple``
        and :class:`numpy.ndarray`.

    Returns
    -------
    bool
        ``True`` when *value* is either a numeric scalar (``int``, ``float``,
        ``numpy.number``) or a homogeneous sequence of numeric scalars;
        ``False`` otherwise.
    """
    if np.isscalar(value):  # int, float, np.number …
        return True

    if isinstance(value, (list, tuple, np.ndarray)):
        # flatten first to support ragged n‑d arrays
        return all(np.isscalar(v) for v in np.ravel(value))

    return False



def _flatten(value: Any) -> np.ndarray:
    """Convert a numeric scalar/sequence into a 1‑D ``float32`` array.

    Non‑numeric objects yield an **empty** array so that callers can decide how
    to handle invalid inputs.

    Parameters
    ----------
    value : Any
        Numeric scalar or sequence (*list*, *tuple*, *ndarray*).

    Returns
    -------
    numpy.ndarray
        One‑dimensional ``float32`` representation of *value*.
    """
    if np.isscalar(value):
        return np.array([value], dtype=np.float32)

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32).ravel()

    # not numeric –> empty vector so feature concatenation still works
    return np.empty(0, dtype=np.float32)
