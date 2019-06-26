# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import math
from mxnet import nd, autograd
from mxnet.test_utils import assert_almost_equal, random_arrays, rand_shape_nd, same
from common import with_seed
import mxnet.autograd as ag
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet
from nose.tools import ok_
import numpy as np


@with_seed()
def test_sin():
    def sin(x):
        return nd.sin(x)

    def grad_grad_op(x):
        return -nd.sin(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sin, grad_grad_op)


@with_seed()
def test_cos():
    def cos(x):
        return nd.cos(x)

    def grad_grad_op(x):
        return -nd.cos(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, cos, grad_grad_op)


@with_seed()
def test_relu():
    def relu(x):
        return nd.relu(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, relu, grad_grad_op)


@with_seed()
def test_log():
    def log(x):
        return nd.log(x)

    def grad_grad_op(x):
        return -1/(x**2)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log, grad_grad_op)


@with_seed()
def test_log2():
    def log2(x):
        return nd.log2(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(2))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log2, grad_grad_op)


@with_seed()
def test_log10():
    def log10(x):
        return nd.log10(x)

    def grad_grad_op(x):
        return -1/((x**2) * math.log(10))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, log10, grad_grad_op)


@with_seed()
def test_reciprocal():
    def reciprocal(x):
        return nd.reciprocal(x)

    def grad_grad_op(x):
        return 2 / x**3

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, reciprocal, grad_grad_op)


@with_seed()
def test_abs():
    def abs(x):
        return nd.abs(x)

    def grad_grad_op(x):
        return nd.zeros_like(x)

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, abs, grad_grad_op)


def test_sigmoid():
    def sigmoid(x):
        return nd.sigmoid(x)

    def grad_op(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def grad_grad_op(x):
        return grad_op(x) * (1 - 2 * sigmoid(x))

    for dim in range(1, 5):
        shape = rand_shape_nd(dim)
        array = random_arrays(shape)
        check_second_order_unary(array, sigmoid, grad_grad_op)


def check_second_order_unary(x, op, grad_grad_op):
    x = nd.array(x)
    grad_grad_x = grad_grad_op(x)
    x.attach_grad()

    # Manual head_grads.
    y_grad = nd.random.normal(shape=x.shape)
    head_grad_grads = nd.random.normal(shape=x.shape)

    # Perform compute.
    with autograd.record():
        y = op(x)
        x_grad = autograd.grad(heads=y, variables=x, head_grads=y_grad,
                               create_graph=True, retain_graph=True)[0]
    x_grad.backward(head_grad_grads)

    # Compute expected values.
    expected_grad_grad = grad_grad_x.asnumpy() * head_grad_grads.asnumpy() * \
        y_grad.asnumpy()

    # Validate the gradients.
    assert_almost_equal(expected_grad_grad, x.grad.asnumpy())

class RandomShapes(object):
    def __init__(self, dim, startdim=1):
        self.dim = dim
        self.curdim = startdim

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.curdim > self.dim:
            raise StopIteration
        shape = rand_shape_nd(self.curdim)
        x = nd.random.normal(shape=shape)
        self.curdim += 1
        return x


@with_seed()
def test_dense_backward():
    for x in RandomShapes(4,2):
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(1))

        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with ag.record():
            y = net.forward(x)
            x_grad = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
        x_grad.backward()
        same(x.grad, nd.zeros(4))

        with ag.record():
            y = net.forward(x)
            x_grad = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
            random_multiplier = nd.random.uniform_like(x_grad)
            z = (random_multiplier * x_grad).sum()
        z.backward()
        same(x.grad, nd.zeros(4))

        with ag.record():
            y = net.forward(x)
            x_grad_0 = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
        x_grad_grad_0 = x.grad

        w_0 = list(net.collect_params().values())[0].data()
        h_w = nd.ones_like(w_0) * 0.01
        net.initialize(mxnet.initializer.Constant(w_0 + h_w), force_reinit=True)
        w_1 = list(net.collect_params().values())[0].data()
        with ag.record():
            y = net.forward(x)
            x_grad_1 = ag.grad(heads=y, variables=x, create_graph=True, retain_graph=True)[0]
        x_grad_1.backward()
        x_grad_grad_1 = x.grad
        ok_(not np.array_equal(x_grad_0, x_grad_1))
        ok_(np.array_equal(x_grad_grad_0, x_grad_grad_1))

        w = list(net.collect_params().values())[0].data()
        with ag.record():
            y = net.forward(x)
            w_grad_0 = ag.grad(heads=y, variables=w, create_graph=True, retain_graph=True)[0]
        w_grad_0.backward()
        w_grad_grad_0 = w.grad

        x = x + nd.ones_like(x) * 0.01
        with ag.record():
            y = net.forward(x)
            w_grad_1 = ag.grad(heads=y, variables=w, create_graph=True, retain_graph=True)[0]
        w_grad_1.backward()
        w_grad_grad_1 = w.grad
        ok_(not np.array_equal(w_grad_0, w_grad_1))
        ok_(np.array_equal(w_grad_grad_0, w_grad_grad_1))



if __name__ == '__main__':
    import nose
    nose.runmodule()
