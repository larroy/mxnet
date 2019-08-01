  /*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file initialize.cc
 * \brief initialize mxnet library
 */
#include <signal.h>
#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include "./engine/openmp.h"
#include "./operator/custom/custom-inl.h"
#if MXNET_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif  // MXNET_USE_OPENCV

namespace mxnet {
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
static void SegfaultLogger(int sig) {
  fprintf(stderr, "\nSegmentation fault: %d\n\n", sig);
  fprintf(stderr, "%s", dmlc::StackTrace().c_str());
  exit(-1);
}
#endif

class LibraryInitializer {
 public:
  static LibraryInitializer* Get();
  LibraryInitializer() {
    dmlc::InitLogging("mxnet");
    install_sigsev_handler();
    set_opencv_concurrency();
  }


 private:
  /**
   * Install segmentation fault handler to print backtraces if enabled at compile time.
   */
  void install_sigsev_handler() {
#if MXNET_USE_SIGNAL_HANDLER && DMLC_LOG_STACK_TRACE
    struct sigaction sa;
    sigaction(SIGSEGV, nullptr, &sa);
    if (sa.sa_handler == nullptr) {
        signal(SIGSEGV, SegfaultLogger);
    }
#endif
  }

  /**
   * Sets the number of threads for opencv given the environment
   */
  void set_opencv_concurrency() {
#if MXNET_USE_OPENCV && !__APPLE__
    const size_t mp_cv_num_threads = dmlc::GetEnv("MXNET_MP_OPENCV_NUM_THREADS", 0);
    cv::setNumThreads(mp_cv_num_threads);  // disable opencv threading
#endif  // MXNET_USE_OPENCV
  }

};

LibraryInitializer* LibraryInitializer::Get() {
  static LibraryInitializer inst;
  return &inst;
}

#ifdef __GNUC__
// Don't print an unused variable message since this is intentional
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static LibraryInitializer* __library_init = LibraryInitializer::Get();
}  // namespace mxnet
