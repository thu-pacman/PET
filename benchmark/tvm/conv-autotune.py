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
.. _opt-conv-gpu:

How to optimize convolution on GPU
==================================
**Author**: `Haichen Shen <https://homes.cs.washington.edu/~haichen/>`_

In this tutorial, we will demonstrate how to write a high performance
convolution implementation in TVM. We use square size input tensors and filters
as an example, and assume the input to convolution has a large batch. In this
example, we use a different layout to store the data in order to achieve better
data locality. The buffer layout is HWCN, which stands for height, width,
channel, batch.

"""

################################################################
# Preparation and Algorithm
# -------------------------
#
# We use the fixed size for input tensors with 256 channels and 14 x 14
# dimensions. The batch size is 256. Convolution filters contain 512 filters
# of size 3 x 3.  We use stride size 1 and padding size 1 for the
# convolution. The following code defines the convolution algorithm in TVM.
#

import numpy as np
import tvm
from tvm import te
import logging
from tvm import autotvm

import sys
# The sizes of inputs and filters
# batch = int(sys.argv[1])
# in_channel = int(sys.argv[2])
# out_channel = int(sys.argv[4])
# in_width = int(sys.argv[3])
# in_height = int(sys.argv[3])
# kernel_width = int(sys.argv[5])
# kernel_height = int(sys.argv[5])
# pad_width = int(sys.argv[6])
# pad_height = int(sys.argv[6])
# stride = int(sys.argv[7])

@autotvm.template('conv-autotune/conv_v1')
def conv2d_autotune(batch, in_channel, out_channel, in_height, in_width, kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width, dtype='float32'):
    # Algorithm
    A = te.placeholder((in_height, in_width, in_channel, batch), name='A', dtype=dtype)
    W = te.placeholder((kernel_height, kernel_width, in_channel, out_channel), name='W', dtype=dtype)
    out_width = (in_width - kernel_width + 2*pad_width) // stride_width + 1
    out_height = (in_height - kernel_height + 2*pad_height) // stride_height + 1
    # Pad input
    Apad = te.compute(
        (in_height + 2*pad_height, in_width + 2*pad_width, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.tir.if_then_else(
            tvm.tir.all(yy >= pad_height, yy - pad_height < in_height,
                    xx >= pad_width, xx - pad_width < in_width),
            A[yy - pad_height, xx - pad_width, cc, nn], tvm.tir.const(0., "float32")),
        name='Apad')
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name='rc')
    ry = te.reduce_axis((0, kernel_height), name='ry')
    rx = te.reduce_axis((0, kernel_width), name='rx')
    # Compute the convolution
    B = te.compute(
        (out_height, out_width, out_channel, batch),
        lambda yy, xx, ff, nn: te.sum(
            Apad[yy * stride_height + ry, xx * stride_width + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')


    ###############################################################################
    # Memory Hierarchy
    # ----------------
    #
    # We first specify the memory hierarchy for buffers. The figure below shows the
    # GPU memory hierarchy. One important difference from CPU memory hierarchy is
    # that GPU provides a cache buffer called shared memory, which is managed by
    # programmers. Thus how to maximize the data reuse in the shared memory is
    # critical to achieve high performance in GPU kernels.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/gpu_memory_hierarchy.png
    #      :align: center
    #      :height: 319px
    #      :width: 271px
    #
    # In this example, we load both Apad and W into buffer AA and WW, which are
    # stored in the shared memory. These bufferes will be later shared by all
    # threads within the same thread block to compute the convolution. Each thread
    # then loads its own part from shared buffer into their local registers, AL and
    # WL. BL is a local cache of output B, which is also stored in the thread local
    # registers.
    #

    # Designate the memory hierarchy
    s = te.create_schedule(B.op)
    s[Apad].compute_inline() # compute Apad inline
    AA = s.cache_read(Apad, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    ###############################################################################
    # Blocking
    # --------
    #
    # The following code splits the workload into thread blocks and individual
    # threads. We follow the blocking scheme in the matrix multiply. As shown in the
    # figure below, given a pixel coordinate (y, x), a thread block is responsible
    # for computing a region of block_factor x block_factor (64 x 64) for output
    # channels and batch. Due to the limit of shared memory space, we only load step
    # x block_factor (8 x 64) data from Apad and B each time to buffers in the
    # shared memory.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/conv_gpu_blocking.png
    #      :align: center
    #      :height: 308px
    #      :width: 317px
    #

    # tile consts
    tile = 4
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    # block_x = te.thread_axis("blockIdx.x")
    # block_y = te.thread_axis("blockIdx.y")
    # block_z = te.thread_axis("blockIdx.z")
    # thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    # thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    # thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    # thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    cfg = autotvm.get_config()
    cfg.define_split('tile_fi', fi, num_outputs=4)
    cfg.define_split('tile_ni', ni, num_outputs=4)
    # cfg.define_split('tile_fi', fi, num_outputs=3)
    # cfg.define_split('tile_ni', ni, num_outputs=3)
    # by, fi = s[B].split(fi, factor=block_factor)
    # bx, ni = s[B].split(ni, factor=block_factor)
    by, tyz, ty, fi = cfg['tile_fi'].apply(s, B, fi)
    bx, txz, tx, ni = cfg['tile_ni'].apply(s, B, ni)
    # by, tyz, ty = cfg['tile_fi'].apply(s, B, fi)
    # bx, txz, tx = cfg['tile_ni'].apply(s, B, ni)

    # Bind the iteration variables to GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    ###############################################################################
    # Virtual Thread Split
    # --------------------
    #
    # We further split the workload from a thread block to individual threads. To
    # avoid *memory bank conflict*, we use virtual thread to split the area into 4
    # parts, and then tile into 8x8 grids. Therefore, shown in the figure below,
    # each thread computes 4 strided grids, where size of each grid is 4 x 4.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/master/tvm/tutorial/conv_gpu_vthread.png
    #      :align: center
    #      :height: 188px
    #      :width: 268px
    #

    # tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    # txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    # ty, fi = s[B].split(fi, nparts=num_thread)
    # tx, ni = s[B].split(ni, nparts=num_thread)
    # thread_y = te.thread_axis((0, ty), "threadIdx.x")
    # thread_x = te.thread_axis((0, tx), "threadIdx.x")
    # thread_xz = te.thread_axis((0, txz), "vthread", name="vx")
    # thread_yz = te.thread_axis((0, tyz), "vthread", name="vy")
    thread_y = te.thread_axis((0, ty), "threadIdx.x")
    thread_x = te.thread_axis((0, tx), "threadIdx.x")
    thread_xz = te.thread_axis((0, txz), "vthread", name="vx")
    thread_yz = te.thread_axis((0, tyz), "vthread", name="vy")
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
    # s[B].reorder(bz, by, bx, tyz, txz, ty, tx)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    ###############################################################################
    # Cooperative Fetching
    # --------------------
    #
    # As mentioned before, each time step we need to transfer step x block_factor
    # data from GPU global memory to shared memory. In order to reduce the memory
    # transfer per thread, the following code lets threads in the same thread block
    # coopertively fetch dependent data from global memory.
    #


    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    # cfg.define_split('tile_ciA', ci, num_outputs=2)
    # cfg.define_split('tile_niA', ni, num_outputs=2)
    # ty, ci = cfg['tile_ciA'].apply(s, AA, ci)
    # tx, ni = cfg['tile_niA'].apply(s, AA, ni)
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    # cfg.define_split('tile_ciW', ci, num_outputs=2)
    # cfg.define_split('tile_fiW', fi, num_outputs=2)
    # ty, ci = cfg['tile_ciW'].apply(s, WW, ci)
    # tx, fi = cfg['tile_fiW'].apply(s, WW, fi)
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fi)  # vectorize memory load
    return s, [A, W, B]


###############################################################################
# Generate CUDA Kernel
# --------------------
#
# Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
# latency of convolution.
#

batch = 128
in_channel = 256
out_channel = 512
in_height, in_width = 14, 14
kernel_height, kernel_width = 3, 3
pad_height, pad_width = 1, 1
stride_height, stride_width = 1, 1
out_width = (in_width - kernel_width + 2*pad_width) // stride_width + 1
out_height = (in_height - kernel_height + 2*pad_height) // stride_height + 1

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

task = autotvm.task.create("conv-autotune/conv_v1",
                           args = (batch, in_channel, out_channel, in_height, in_width, kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width),
                           target = 'cuda')
print(task.config_space)

measure_option = autotvm.measure_option(
    builder = autotvm.LocalBuilder(),
    runner = autotvm.LocalRunner(repeat=3, min_repeat_ms = 100, timeout = 4)
)
tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
tuner.tune(n_trial=200,
           measure_option = measure_option,
           callbacks = [autotvm.callback.log_to_file('conv2d.log')])

dispatch_context = autotvm.apply_history_best('conv2d.log')
best_config = dispatch_context.query(task.target, task.workload)
print('\nBest config:')
print(best_config)

with autotvm.apply_history_best('conv2d.log'):
    with tvm.target.create('cuda'):
        s, arg_bufs = conv2d_autotune(batch, in_channel, out_channel, in_height, in_width, kernel_height, kernel_width, pad_height, pad_width, stride_height, stride_width)
        func = tvm.build(s, arg_bufs)

ctx = tvm.gpu(0)
dtype = 'float32'
a_np = np.random.uniform(size=(in_height, in_width, in_channel, batch)).astype(dtype)
w_np = np.random.uniform(size=(kernel_height, kernel_width, in_channel, out_channel)).astype(dtype)
a = tvm.nd.array(a_np, ctx)
w = tvm.nd.array(w_np, ctx)
b = tvm.nd.array(np.zeros((out_height, out_width, out_channel, batch), dtype=dtype), ctx)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Convolution: %f ms' % (evaluator(a, w, b).mean * 1e3))
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Convolution: %f ms' % (evaluator(a, w, b).mean * 1e3))
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Convolution: %f ms' % (evaluator(a, w, b).mean * 1e3))
# dev_module = func.imported_modules[0]
# print(dev_module)
# print("----GPU code----")
# print(dev_module.get_source())
