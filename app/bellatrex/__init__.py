# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:19:54 2024
bellatrex package initializer.
@author: Klest Dedja
"""

from typing import TYPE_CHECKING  # so that pylance checker does not complain

try:
    from .__version__ import __version__
except ImportError:
    __version__ = "unknown"


# Lazy import to safely expose BellatrexExplain to
# outer layer only after (ediatble) installation is complete
def __getattr__(name):
    if name == "BellatrexExplain":
        from .LocalMethod_class import BellatrexExplain  # pylint: disable=import-outside-toplevel

        return BellatrexExplain
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["BellatrexExplain"]
