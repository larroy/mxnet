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
import random
from functools import reduce
from operator import mul
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

def arange_shape_like(y):
    shape = y.shape
    nelems = reduce(mul, shape)
    x = nd.arange(nelems).reshape(shape)
    return x

class RandomShapes(object):
    def __init__(self, dim, startdim=1):
        self.dim = dim
        self.curdim = startdim

    def __iter__(self):
        return self

    @staticmethod
    def random_shape(dimensions):
        shape = rand_shape_nd(dimensions)
        # x = nd.random.normal(shape=shape)
        nelems = reduce(mul, shape)
        x = nd.arange(nelems).reshape(shape)
        return x

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.curdim > self.dim:
            raise StopIteration
        x = RandomShapes.random_shape(self.curdim)
        self.curdim += 1
        return x


def flatten2d_right(x):
    s_0 = x.shape[0]
    s_1 = reduce(mul, x.shape[1:])
    return x.reshape((s_0, s_1))


def flatten2d_left(x):
    s_0 = reduce(mul, x.shape[:-1])
    s_1 = x.shape[-1]
    return x.reshape((s_0, s_1))


@with_seed()
def test_dense_backward_flatten():
    for x in RandomShapes(4,2):
        hidden = random.randrange(1, 4)
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(hidden, flatten=True))
        net.initialize(mxnet.initializer.Constant(.5))
        x.attach_grad()
        with ag.record():
            y = net.forward(x)
            o_y = arange_shape_like(y)  # head gradient of y
            params = [p.data() for p in net.collect_params().values()]
            w = params[0]
            b = params[1]

            # print(params)
            x_grad = ag.grad(heads=y, variables=x, head_grads=o_y,
                             create_graph=True, retain_graph=True)[0]
            o_x_grad = arange_shape_like(x_grad)
            #x_grad.attach_grad()
            x_grad_grad = ag.grad(heads=x_grad, variables=w, head_grads=o_x_grad, create_graph=False)[0]
            w_grad = ag.grad(heads=y, variables=w, head_grads=o_y,
                             create_graph=True, retain_graph=True)[0]
            o_w_grad = arange_shape_like(w_grad)
            w_grad_grad = ag.grad(heads=w_grad, variables=x, head_grads=o_w_grad, create_graph=False)[0]

        expect_w_grad = nd.dot(o_y, x, transpose_a=True)
        expect_w_grad_grad = nd.dot(o_y, o_x_grad, transpose_a=True)
        expect_x_grad = nd.dot(o_y, w)
        expect_x_grad_grad = nd.dot(o_y, o_w_grad)
        same(expect_w_grad, w_grad)
        same(expect_w_grad_grad, w_grad_grad)
        same(expect_x_grad, x_grad)
        same(expect_x_grad_grad, x_grad_grad)


if __name__ == '__main__':
    import nose
    nose.runmodule()
