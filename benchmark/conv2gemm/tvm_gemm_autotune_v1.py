import tvm
from tvm import te
from tvm import autotvm
import numpy as np
import logging
import sys

@autotvm.template('gemm-autotune/gemm_v1')
def gemm_autotune(m, n, k, dtype='float32'):
    A = te.placeholder((m,k), name='A', dtype=dtype)
    B = te.placeholder((k,n), name='B', dtype=dtype)
    kk = te.reduce_axis((0, k), name='kk')
    C = te.compute(
            (m, n), lambda mm, nn: te.sum(A[mm,kk]*B[kk,nn], axis=[kk]),
            name='C')

    s = te.create_schedule(C.op)
    AL = s.cache_read(A, 'local', [C])
    BL = s.cache_read(B, 'local', [C])
    CL = s.cache_write(C, 'local')

    axm, axn = s[C].op.axis
    cfg = autotvm.get_config()
    # cfg.define_split('tile_m', axm, num_outputs=5)
    # cfg.define_split('tile_n', axn, num_outputs=5)
    # bx, by, tz, txz, mi = cfg['tile_m'].apply(s, C, axm)
    # tx, ty, bz, tyz, ni = cfg['tile_n'].apply(s, C, axn)
    cfg.define_split('tile_m', axm, num_outputs=4)
    cfg.define_split('tile_n', axn, num_outputs=4)
    bx, by, tz, mi = cfg['tile_m'].apply(s, C, axm)
    tx, ty, bz, ni = cfg['tile_n'].apply(s, C, axn)
    # axkk = s[C].op.reduce_axis
    # cfg.define_split('tile_k', axkk, num_outputs=3)
    # kx, ky, ki = cfg['tile_k'].apply(s, C, axkk)

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    block_z = te.thread_axis('blockIdx.z')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')
    # thread_xz = te.thread_axis((0, txz), 'vthread', name='vx')
    # thread_yz = te.thread_axis((0, tyz), 'vthread', name='vy')

    s[C].bind(tx, thread_x)
    s[C].bind(ty, thread_y)
    s[C].bind(tz, thread_z)
    s[C].bind(bx, block_x)
    s[C].bind(by, block_y)
    s[C].bind(bz, block_z)
    # s[C].bind(txz, thread_xz)
    # s[C].bind(tyz, thread_yz)

    # s[C].reorder(bz, by, bx, tyz, txz, tz, ty, tx, mi, ni)
    s[C].reorder(bz, by, bx, tz, ty, tx, mi, ni)

    s[CL].compute_at(s[C], tx)
    caxm, caxn = s[CL].op.axis
    cbx, cby, ctz, cmi = cfg['tile_m'].apply(s, CL, caxm)
    ctx, cty, cbz, cni = cfg['tile_n'].apply(s, CL, caxn)
    # cbx, cby, ctz, ctxz, cmi = cfg['tile_m'].apply(s, CL, caxm)
    # ctx, cty, cbz, ctyz, cni = cfg['tile_n'].apply(s, CL, caxn)
    s[CL].reorder(cbz, cby, cbx, ctz, cty, ctx, cmi, cni)

    s[AL].compute_at(s[C], tx)
    aaxm, akk = s[AL].op.axis
    abx, aby, atz, ami = cfg['tile_m'].apply(s, AL, aaxm)
    # abx, aby, atz, atxz, ami = cfg['tile_m'].apply(s, AL, aaxm)
    s[AL].reorder(aby, abx, atz, ami)
    _, akki = s[AL].split(akk, factor=4)
    s[AL].vectorize(akki)

    s[BL].compute_at(s[C], tx)
    bkk, baxn = s[BL].op.axis
    btx, bty, bbz, bni = cfg['tile_n'].apply(s, BL, baxn)
    # btx, bty, bbz, btyz, bni = cfg['tile_n'].apply(s, BL, baxn)
    s[BL].reorder(bbz, bty, btx, bni)
    _, bkki = s[BL].split(bkk, factor=4)
    s[BL].vectorize(bkki)

    return s, [A, B, C]

m, n, k = 4096, 4096, 512

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

task = autotvm.task.create('gemm-autotune/gemm_v1',
                           args = (m, n, k),
                           target = 'cuda')
print(task.config_space)

measure_option = autotvm.measure_option(
    builder = autotvm.LocalBuilder(),
    runner = autotvm.LocalRunner(repeat=3, min_repeat_ms = 100, timeout = 4)
)
tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
tuner.tune(n_trial=200,
           measure_option = measure_option,
           callbacks = [autotvm.callback.log_to_file('gemm.log')])

dispatch_context = autotvm.apply_history_best('gemm.log')
best_config = dispatch_context.query(task.target, task.workload)
print('\nBest config:')
print(best_config)


