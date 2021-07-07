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
"""
.. _auto-scheduler-conv-gpu:

Auto-scheduling a Convolution Layer for GPU
===========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

This is a tutorial on how to use the auto-scheduler for GPUs.

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any templates.
Users only need to write the computation declaration without any schedule commands or templates.
The auto-scheduler can automatically generate a large search space and
find a good schedule in the space.

We use a convolution layer as an example in this tutorial.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import os
import sys

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi, autotvm
from tvm.topi.testing import conv2d_nchw_python

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a convolution layer.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding, dilation):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(
        data, kernel, stride, padding, dilation=dilation, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


@auto_scheduler.register_workload
def bmm_layer(M, N, K, B):
    A = te.placeholder((B, M, K), name="A")
    B = te.placeholder((B, N, K), name="B")
    out = topi.nn.batch_matmul(A, B)
    return [A, B, out]


######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task for the last convolution layer in the resnet.

target = tvm.target.Target("cuda")

# Use the last layer in ResNet-50
# N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
# task = auto_scheduler.create_task(conv2d_layer, (N, H, W, CO, CI, KH, KW, strides, padding), target)
tasks = []
# N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 512, 1024, 3, 3, (1, 1), (1, 1)
# log_file = "32x4d-Conv2d-104.json"
# N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 512, 1024, 1, 1, (1, 1), (1, 1)
# log_file = "CSRNet-dilated-1.json"
# N, H, W, CO, CI, KH, KW, strides, padding, dilation = 1, 14, 14, 512, 512, 3, 3, (1, 1), (1, 1), 4
# log_file = "CSRNet-dilated-2.json"
# N, H, W, CO, CI, KH, KW, strides, padding, dilation = 1, 14, 14, 128, 512, 3, 3, (1, 1), (1, 1), 4
log_file = "bmm-1-nico1.json"

# https://docs.google.com/spreadsheets/d/1it9z3boqPLeKhjUmVgAX3WWuqJ5aBpUbg5xhOlrWVYc/edit#gid=1102136610
input_tasks = [  # Input, Kernel, p, s, d
    # Conv-2	-, 128, 28, 28	128, 128, 3,3
    # [[1, 128, 28, 28], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 128, 14, 56], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 128, 56, 14], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 128, 7, 112], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 128, 112, 7], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[2, 128, 14, 28], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[2, 128, 28, 14], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 128, 14, 14], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 128, 7, 28], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 128, 28, 7], [128, 128, 3, 3], [1, 1], [1, 1], [1, 1]],
    # # Conv new added [ 1, 48, 38, 38], weight = [64, 48, 5, 5], p = [2, 2], s = [1, 1], d = [1, 1],
    # [[1, 48, 38, 38], [64, 48, 5, 5], [2, 2], [1, 1], [1, 1]],
    # [[16, 48, 10, 10], [64, 48, 5, 5], [2, 2], [1, 1], [1, 1]],
    # # merge to a group conv
    # [[1, 768, 18, 18], [192, 768, 1, 1], [0, 0], [1, 1], [1, 1]],
    # [[1, 768, 18, 18], [160, 768, 1, 1], [0, 0], [1, 1], [1, 1]],
    # Conv-3 Conv-4
    # -, 256, 14, 14	256, 256, 3, 3
    # -, 512, 7, 7	512, 512, 3, 3
    # [[1, 256, 14, 14], [256, 256, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 256, 14, 14], [256, 256, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[16, 256, 14, 14], [256, 256, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[64, 256, 14, 14], [256, 256, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 512, 7, 7], [512, 512, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 512, 7, 7], [512, 512, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[16, 512, 7, 7], [512, 512, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[64, 512, 7, 7], [512, 512, 3, 3], [1, 1], [1, 1], [1, 1]],
    # # Conv-1 mutant -, 64, 56, 56	64, 64, 3, 3
    # [[1, 64, 56, 56], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 64, 112, 28], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 64, 28, 112], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 64, 224, 14], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[1, 64, 14, 224], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[2, 64, 28, 56], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[2, 64, 56, 28], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 64, 28, 28], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 64, 14, 56], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # [[4, 64, 56, 14], [64, 64, 3, 3], [1, 1], [1, 1], [1, 1]],
    # # Conv9变形 [ 1, 48, 38, 38]	[64, 48, 5, 5]
    # [[1, 48, 38, 38], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[1, 48, 76, 19], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[1, 48, 19, 76], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[2, 48, 19, 38], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[2, 48, 38, 19], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[4, 48, 19, 19], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # [[16, 48, 10, 10], [64, 48, 5, 5], [1, 1], [1, 1], [1, 1]],
    # # Dialated Conv7 变形
    # [[1, 512, 14, 14],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[1, 512, 28, 7],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[1, 512, 7, 28],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[2, 512, 7, 14],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[2, 512, 14, 7],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[4, 512, 7, 7],  [256, 512, 3, 3],  [2, 2], [1, 1], [2, 2]],
    # [[1, 512, 14, 14],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    # [[1, 512, 28, 7],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    # [[1, 512, 7, 28],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    # [[2, 512, 7, 14],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    # [[2, 512, 14, 7],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    # [[4, 512, 7, 7],  [256, 512, 3, 3],  [1, 1], [1, 1], [1, 1]],
    [512, 768, 768, 1],
    # [512, 768, 768, 3],
]

for input_task in input_tasks:
    M, N, K, B = input_task
    # task = auto_scheduler.create_task(
    #     conv2d_layer, (N, H, W, CO, CI, KH, KW, strides, padding, dilation), target)
    task = auto_scheduler.create_task(bmm_layer, (M, N, K, B), target)
    tasks.append(task)

# for input_task in input_tasks:
#     [[N, CI, H, W], [CO, _, KH, KW], padding, strides, dilation] = input_task
#     assert(CI == _)
#     task = auto_scheduler.create_task(
#         conv2d_layer, (N, H, W, CO, CI, KH, KW, strides, padding, dilation), target)
#     tasks.append(task)


print('# of tasks = %d' % (len(tasks)))
# Inspect the computational graph
# print(task.compute_dag)

######################################################################
# Next, we set parameters for the auto-scheduler. These parameters
# mainly specify how we do the measurement during the search.
#
# * :code:`measure_ctx` launches a different process for measurement to
#   provide isolation. It can protect the master process from GPU crashes
#   during measurement and avoid other runtime conflicts.
# * :code:`min_repeat_ms` defines the minimum duration of one "repeat" in every measurement.
#   This can warmup the GPU, which is necessary to get accurate measurement results.
#   Typically, we recommend a value > 300 ms.
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file `conv2d.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRPCMeasureContext` for more parameters.

measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    # change this to 1000 to achieve the best performance
    num_measure_trials=1024*len(tasks)+1,
    # runner=measure_ctx.runner,
    runner=auto_scheduler.RPCRunner(
        "nico2_v100_32",  # change the device key to your key
        "0.0.0.0",
        9190,
        n_parallel=7,
        number=5,
        repeat=1,
        timeout=20,
        min_repeat_ms=300,
    ),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)


print("Begin tuning multiple bmms...")
# measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400, timeout=10)
tuner = auto_scheduler.TaskScheduler(tasks, strategy='round-robin')
tuner.tune(tune_option)
sys.exit()
