import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
from tvm import autotvm
import sys
import logging

@autotvm.template('gemm-autotune/gemm_v2')
def gemm_autotune(mm, nn, ll, dtype='float32'):
    # graph
    m, n, l = te.var('m'), te.var('n'), te.var('l')
    m, n, l = tvm.runtime.convert(mm), tvm.runtime.convert(nn), tvm.runtime.convert(ll)
    A = te.placeholder((l, n), name='A', dtype=dtype)
    B = te.placeholder((l, m), name='B', dtype=dtype)
    k = te.reduce_axis((0, l), name='k')
    C = te.compute(
        (m, n),
        lambda ii, jj: te.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    # schedule
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    cfg = autotvm.get_config()
    cfg.define_knob('tile_parts', [2**i for i in range(5)])
    cfg.define_knob('tile_num_thread', [2**i for i in range(10)])
    cfg.define_knob('tile_num_block', [2**i for i in range(10)])
    # scale = 8
    # num_thread = 8
    parts = cfg['tile_parts'].val
    num_threadx = cfg['tile_num_thread'].val
    num_thready = cfg['tile_num_thread'].val
    block_factorx = cfg['tile_num_block'].val
    block_factory = cfg['tile_num_block'].val
    # block_factor = scale * num_thread
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_threadx), "threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis((0, num_thready), "threadIdx.y")
    thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

    by, yi = s[C].split(C.op.axis[0], factor=block_factory)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factorx)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)

    tyz, yi = s[C].split(yi, nparts=parts)
    ty, yi = s[C].split(yi, nparts=num_thready)
    txz, xi = s[C].split(xi, nparts=parts)
    tx, xi = s[C].split(xi, nparts=num_threadx)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    ko, ki = s[CC].split(k, factor=8)
    kt, ki = s[CC].split(ki, factor=1)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)
    # Schedule for A's shared memory load
    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thready)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_threadx * 4)
    tx, xi = s[AA].split(xi, nparts=num_threadx)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(xi)
    # Schedule for B' shared memory load
    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thready)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_threadx * 4)
    tx, xi = s[BB].split(xi, nparts=num_threadx)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s[BB].vectorize(xi)
    s[AA].double_buffer()
    s[BB].double_buffer()

    return s, [A, B, C]

def test_gemm(mm, nn, ll):
    # correctness
    m, n, l = mm, nn, ll
    dtype = 'float32'

    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    log_file = 'gemm.log'

    task = autotvm.task.create('gemm-autotune/gemm_v2',
                               args = (m, n, l), target='cuda')
    print(task.config_space)

    measure_option = autotvm.measure_option(
        builder = autotvm.LocalBuilder(),
        runner = autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )
    tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
    tuner.tune(n_trial=1000,
               measure_option = measure_option,
               callbacks = [autotvm.callback.log_to_file(log_file)])

    dispatch_context = autotvm.apply_history_best(log_file)
    best_config = dispatch_context.query(task.target, task.workload)
    print('\nBest config:')
    print(best_config)

    with autotvm.apply_history_best(log_file):
        with tvm.target.create('cuda'):
            s, arg_bufs = gemm_autotune(m, n, l)
            f = tvm.build(s, arg_bufs)
    # launch the kernel.
    # a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
    # b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
    ctx = tvm.gpu(0)
    a_np = np.random.uniform(size=(l, n)).astype(dtype)
    b_np = np.random.uniform(size=(l, m)).astype(dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((m, n), dtype=dtype), ctx)
    for i in range(2):
        f(a, b, c)
    print('function called')
    tvm.testing.assert_allclose(
        c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e-5)

    num_flops = 2 * nn * mm * ll
    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e9
    print("average time cost of %d runs = %g ms, %g TFLOPS." % (num_runs, t * 1e3, GFLOPS))

    # for device in ["cuda", "opencl", "rocm", "nvptx", "vulkan"]:
    #     with tvm.transform.PassContext(config={"tir.UnrollLoop": {
    #         "auto_max_step": 128,
    #         "explicit_unroll": device != "cuda"
    #     }}):
    #         check_device(device)

if __name__ == "__main__":
    test_gemm(2048, 256, 1024)
