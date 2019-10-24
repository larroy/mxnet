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
 * \file graph_dump.h
 * \brief Utilities to introspect and print the execution graph
 * \author Pedro Larroy
 */

#ifndef MXNET_NNVM_GRAPH_DUMP_H_
#define MXNET_NNVM_GRAPH_DUMP_H_

#include <vector>
#include <string>
#include "nnvm/node.h"

namespace nnvm {

/*!
 * Returns a human readable graph
 */
std::string GraphDump(const std::vector<NodeEntry>& graph);

}  // end namespace nnvm
#endif  // MXNET_NNVM_GRAPH_DUMP_H _
