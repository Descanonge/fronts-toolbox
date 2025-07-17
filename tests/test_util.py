"""Test utility functions."""

import dask.array as da
import numba.types as nt
import numpy as np
import pytest
import xarray as xr
from numba import guvectorize
from numpy.testing import assert_allclose

from fronts_toolbox.util import (
    Dispatcher,
    apply_vectorized,
    get_axes_kwarg,
    get_window_reach,
)

rng = np.random.default_rng()


def test_window_reach():
    assert get_window_reach(3) == [1, 1]
    assert get_window_reach(5) == [2, 2]
    assert get_window_reach(11) == [5, 5]

    assert get_window_reach((3, 5)) == [1, 2]

    with pytest.raises(ValueError):
        get_window_reach(2)
    with pytest.raises(ValueError):
        get_window_reach((3, 2))


def test_get_axes_kwarg():
    # Basic
    signature = "(y,x)->(y,x)"
    assert get_axes_kwarg(signature, [-2, -1], order="y,x") == [(-2, -1), (-2, -1)]
    assert get_axes_kwarg(signature, [1, 2], order="y,x") == [(1, 2), (1, 2)]

    # Reverse order
    signature = "(y,x)->(y,x)"
    assert get_axes_kwarg(signature, [-2, -1], order="x,y") == [(-1, -2), (-1, -2)]
    assert get_axes_kwarg(signature, [1, 2], order="x,y") == [(2, 1), (2, 1)]

    # Multiple outputs
    signature = "(y,x)->(y,x),(x,y)"
    assert get_axes_kwarg(signature, [-2, -1], order="x,y") == [
        (-1, -2),
        (-1, -2),
        (-2, -1),
    ]
    assert get_axes_kwarg(signature, [1, 2], order="x,y") == [(2, 1), (2, 1), (1, 2)]

    # With additional arguments
    signature = "(y,x),(a,b),(c),(),()->(y,x)"
    assert get_axes_kwarg(signature, [-2, -1], order="y,x") == [
        (-2, -1),
        (0, 0),
        (0,),
        (),
        (),
        (-2, -1),
    ]
    assert get_axes_kwarg(signature, [1, 2], order="y,x") == [
        (1, 2),
        (0, 0),
        (0,),
        (),
        (),
        (1, 2),
    ]


@guvectorize(
    [(nt.float64[:, :], nt.float64[:, :])], "(y,x)->(y,x)", nopython=True, cache=True
)
def ref_gufunc(x, output):
    output[:] = np.mean(x)


def test_apply_vectorized():
    def py_gufunc(x):
        output = np.zeros_like(x)
        output[:] = np.mean(x)
        return output

    # Simple 2D (no vectorization needed)
    x = rng.random((16, 32))
    assert_allclose(ref_gufunc(x), apply_vectorized(py_gufunc, x))
    # reverse axes
    assert_allclose(
        ref_gufunc(x, axes=[(1, 0), (1, 0)]),
        apply_vectorized(py_gufunc, x, axes=[1, 0]),
    )

    # 3D
    x = rng.random((5, 16, 32))
    assert_allclose(ref_gufunc(x), apply_vectorized(py_gufunc, x))
    # reverse axes
    assert_allclose(
        ref_gufunc(x, axes=[(2, 1), (2, 1)]),
        apply_vectorized(py_gufunc, x, axes=[2, 1]),
    )

    # 3D, not last axes
    x = rng.random((16, 5, 32))
    assert_allclose(
        ref_gufunc(x, axes=[(0, 2), (0, 2)]),
        apply_vectorized(py_gufunc, x, axes=[0, 2]),
    )

    # 4D, not last axes
    x = rng.random((5, 16, 2, 32))
    assert_allclose(
        ref_gufunc(x, axes=[(1, 3), (1, 3)]),
        apply_vectorized(py_gufunc, x, axes=[1, 3]),
    )


class TestDispatcher:
    arr_np = np.ones(3)
    arr_da = da.ones(3)
    arr_xr = xr.DataArray(np.ones((3, 3)), dims=("lat", "lon"))

    def func(self, name):
        def f():
            return name

        return f

    def test_dispatch(self):
        d = Dispatcher(
            "",
            numpy=self.func("numpy"),
            dask=self.func("dask"),
            xarray=self.func("xarray"),
        )

        assert d.get_func(self.arr_np)() == "numpy"
        assert d.get_func(self.arr_da)() == "dask"
        assert d.get_func(self.arr_xr)() == "xarray"

    def test_missing_function(self):
        d = Dispatcher("", numpy=self.func("numpy"))
        assert d.get_func(self.arr_np)() == "numpy"

        with pytest.raises(NotImplementedError):
            d.get_func(self.arr_da)
