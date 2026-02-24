"""
vsapy – Vector Symbolic Architecture library

"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("vsapy")
except Exception:  # fallback during local dev
    __version__ = "0.10.0-dev"


# ---------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------

from .vsatype import VsaType, VsaBase

# ---------------------------------------------------------------------
# Concrete VSA Implementations
# ---------------------------------------------------------------------

from .bsc import BSC
from .tern import Tern
from .ternzero import TernZero
from .hrr import HRR
from .laiho import Laiho
from .laihox import LaihoX

# ---------------------------------------------------------------------
# Core Operators / Utilities
# ---------------------------------------------------------------------

from .vsapy import (
    randvec,
    normalize,
    hsim,
    hdist,
    bind,
    unbind,
)

from .bag import *
from .vsa_tokenizer import VsaTokenizer

# ---------------------------------------------------------------------
# Number Line (including new circular variants)
# ---------------------------------------------------------------------

from .number_line import (
    NumberLine,
    linear_sequence_gen,
    CircularNumberLineFolded,
    CircularNumberLinePhaseProjected,
)

# ---------------------------------------------------------------------
# Explicit Public API
# ---------------------------------------------------------------------

__all__ = [
    "__version__",

    # Core types
    "VsaType",
    "VsaBase",

    # Implementations
    "BSC",
    "Tern",
    "TernZero",
    "HRR",
    "Laiho",
    "LaihoX",

    # Core construction
    "randvec",
    "normalize",
    "hsim",
    "hdist",
    "bind",
    "unbind",

    # Tokenizer
    "VsaTokenizer",

    # Number lines
    "NumberLine",
    "linear_sequence_gen",
    "CircularNumberLineFolded",
    "CircularNumberLinePhaseProjected",
]
