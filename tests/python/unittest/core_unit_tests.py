#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core unit tests to run to validate MXNet"""

import os
import sys
import nose

from nose.loader import TestLoader
from nose import run
from nose.suite import LazySuite



def main():
    os.environ['MXNET_STORAGE_FALLBACK_LOG_VERBOSE'] = '0'
    tp = nose.core.TestProgram(defaultTest=[
        'test_ndarray',
        'tets_base',
        'test_engine',
        'test_autograd',
        'test_infer_shape',
        'test_attr',
        'test_engine_import',
        'test_predictor',
        'test_metric',
        'test_rnn',
        'test_symbol',
        'test_subgraph',
        'test_executor',
        'test_module',
        'test_loss'
    ]
    )
    return 0

if __name__ == '__main__':
    sys.exit(main())

