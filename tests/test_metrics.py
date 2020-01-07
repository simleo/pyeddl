# Copyright (c) 2019 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pytest
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT
from pyeddl._core import getCustomMetric


def py_mse(t, y):
    aux = eddlT.add(t, eddlT.neg(y))
    aux = eddlT.mult(aux, aux)
    return aux.sum() / eddlT.getShape(t)[1]


def py_mse_numpy(t, y):
    a = np.array(t, copy=False)
    b = np.array(y, copy=False)
    return np.sum(np.square(a - b)) / a.shape[1]


def test_custom_metric():
    T = eddlT.ones([3, 4])
    Y = eddlT.create([3, 4])
    eddlT.fill_(Y, 0.15)
    exp_v = eddl.getMetric("mse").value(T, Y)
    m = getCustomMetric(py_mse, "py_mse")
    assert pytest.approx(m.value(T, Y), exp_v)
    m2 = getCustomMetric(py_mse_numpy, "py_mse")
    assert pytest.approx(m2.value(T, Y), exp_v)
