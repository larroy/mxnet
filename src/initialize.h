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
 * Copyright (c) 2019 by Contributors
 * \file initialize.h
 * \brief Library initialization
 */

#include <cstdlib>

#ifndef MXNET_INITIALIZE_H_
#define MXNET_INITIALIZE_H_

namespace mxnet {

void pthread_atfork_prepare();
void pthread_atfork_parent();
void pthread_atfork_child();

/**
 * Perform library initialization and control multiprocessing behaviour.
 */
class LibraryInitializer {
 public:
  static LibraryInitializer* Get() {
    static LibraryInitializer inst;
    return &inst;
  }

  /**
   * Library initialization. Called on library loading via constructor attributes or
   * C++ static initialization.
   */
  LibraryInitializer();

  /**
   * @return true if the current pid doesn't match the one that initialized the library
   */
  bool was_forked() const;

  /**
   * Original pid of the process which first loaded and initialized the library
   */
  size_t original_pid_;
  size_t mp_worker_nthreads_;
  size_t cpu_worker_nthreads_;
  size_t omp_num_threads_;
  size_t mp_cv_num_threads_;

  // Actual code for the atfork handlers as member functions.

  void atfork_prepare();
  void atfork_parent();
  void atfork_child();

 private:
  /**
   * Pthread atfork handlers are used to reset the concurrency state of modules like CustomOperator
   * and Engine when forking. When forking only the thread that forks is kept alive and memory is
   * copied to the new process so state is inconsistent. This call install the handlers.
   * Has no effect on Windows.
   *
   * https://pubs.opengroup.org/onlinepubs/009695399/functions/pthread_atfork.html
   */
  void install_pthread_atfork_handlers();

  /**
   * Install signal handlers (UNIX). Has no effect on Windows.
   */
  void install_signal_handlers();
};


}  // namespace mxnet
#endif  // MXNET_INITIALIZE_H_
