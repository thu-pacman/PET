import numpy as np
import tvm
from tvm import te

batch = 128
in_channel = 3
in_size = 224

# nhwc
A = te.placeholder((batch, in_channel, in_size, in_size), name='A')

A_ch = te.compute(
    (batch, in_channel, in_size, in_size),
    lambda n, c, h, w: A[n, c, h, w],
    name='A_change')

s = te.create_schedule(A_ch.op)

blockdim, threaddim = 32, 32
block_x = te.thread_axis((0, blockdim), "blockIdx.x")
block_y = te.thread_axis((0, blockdim), "blockIdx.y")
thread_x = te.thread_axis((0, threaddim), "threadIdx.x")
thread_y = te.thread_axis((0, threaddim), "threadIdx.y")

n, c, h, w = s[A_ch].op.axis
no, ni = s[A_ch].split(n, nparts=blockdim)
co, ci = s[A_ch].split(c, nparts=blockdim)
ho, hi = s[A_ch].split(h, nparts=threaddim)
wo, wi = s[A_ch].split(w, nparts=threaddim)
s[A_ch].bind(no, block_y)
s[A_ch].bind(co, block_x)
s[A_ch].bind(ho, thread_y)
s[A_ch].bind(wo, thread_x)
wio, wii = s[A_ch].split(wi, factor=4)
s[A_ch].reorder(no, co, ho, wo, ni, ci, hi, wio, wii)
s[A_ch].vectorize(wii)
print(tvm.lower(s, [A, A_ch], simple_mode=True))

func = tvm.build(s, [A, A_ch], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(batch, in_channel, in_size, in_size)).astype(A.dtype)
a = tvm.nd.array(a_np, ctx)
a_ch = tvm.nd.array(np.zeros((batch, in_channel, in_size, in_size), dtype=A_ch.dtype), ctx)
func(a, a_ch)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
conv_time = evaluator(a, a_ch).mean * 1e3
tot_byte = batch * in_channel * in_size * in_size * 4 / 1024 / 1024 / 1024 # GB
print('Convolution: %f ms, Bandwidth: %f GB/s' % (conv_time, tot_byte / conv_time * 1000))

dev_module = func.imported_modules[0]
print(dev_module)
print("----GPU code----")
print(dev_module.get_source())
