import numpy as np
import tvm
from tvm import te

batch = 128
in_channel = 64
in_size = 64

# chwn -> nchw
input_tensor = (in_channel, in_size, in_size, batch)
output_tensor = (batch, in_channel, in_size, in_size)

A = te.placeholder(input_tensor, name='A')

A_ch = te.compute(
    output_tensor,
    lambda n, c, h, w: A[c, h, w, n], # modification
    name='A_change')

s = te.create_schedule(A_ch.op)

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")


blockdim, threaddim = 32, 32
n, c, h, w = s[A_ch].op.axis
no, ni = s[A_ch].split(n, factor=blockdim)
co, ci = s[A_ch].split(c, factor=blockdim)
ho, hi = s[A_ch].split(h, factor=threaddim)
wo, wi = s[A_ch].split(w, factor=threaddim)
s[A_ch].bind(ni, block_y)
s[A_ch].bind(ci, block_x)
s[A_ch].bind(hi, thread_y)
s[A_ch].bind(wi, thread_x)
s[A_ch].reorder(no, co, ho, wo, ni, ci, hi, wi)
hwo = s[A_ch].fuse(ho, wo)
# s[A_ch].vectorize(hwo)
# s[A_ch].reorder(co, ho, wo, no, ci, hi, wi, ni)
print(tvm.lower(s, [A, A_ch], simple_mode=True))
"""
blockdim, threaddim = 32, 32
n, c, h, w = s[A_ch].op.axis
hw = s[A_ch].fuse(h, w)
no, ni = s[A_ch].split(n, nparts=blockdim)
co, ci = s[A_ch].split(c, nparts=blockdim)
hwo, hwi = s[A_ch].split(hw, nparts=32*32)
s[A_ch].reorder(no, co, hwo, ni, ci, hwi)
s[A_ch].bind(no, block_y)
s[A_ch].bind(co, block_x)
s[A_ch].bind(hwo, thread_x)
s[A_ch].vectorize(hwi)
"""          
func = tvm.build(s, [A, A_ch], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=input_tensor).astype(A.dtype)
a = tvm.nd.array(a_np, ctx)
a_ch = tvm.nd.array(np.zeros(output_tensor, dtype=A_ch.dtype), ctx)
func(a, a_ch)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
conv_time = evaluator(a, a_ch).mean * 1e3
tot_byte = batch * in_channel * in_size * in_size * 4 / 1024 / 1024 / 1024 # GB
print('Convolution: %f ms, Bandwidth: %f GB/s' % (conv_time, tot_byte / conv_time * 1000 * 2))

dev_module = func.imported_modules[0]
print(dev_module)
print("----GPU code----")
print(dev_module.get_source())

