"""Belkin-O'Reilly Algorithm filter."""

from __future__ import annotations

import logging
from collections.abc import Collection, Hashable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numba import guvectorize, jit, prange

from fronts_toolbox.util import FuncMapper

if TYPE_CHECKING:
    from fronts_toolbox.util import DaskArray, NDArray, XarrayArray
