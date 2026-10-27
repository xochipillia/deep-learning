"""Small shared statistics helpers (no third-party dependencies)."""
from __future__ import annotations

import json
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, as fractions in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(c - h, 6), round(c + h, 6))


def paper_reference() -> dict:
    """Reference numbers from the paper / official README (full validation, n=4,651)."""
    return json.loads((Path(__file__).resolve().parent / "paper_reference.json").read_text())
