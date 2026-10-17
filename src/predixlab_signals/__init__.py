"""PredixLab Signals — a lightweight swing-point prediction model.

The package answers one question per bar: *is this price a local low (buy) or a
local high (sell), and how much room is there in either direction?*  It does so
with a small, scale-free feature set and three compact LightGBM heads.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
