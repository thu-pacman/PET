import numpy as np
import tvm
from tvm import te

batch = 128
in_channel = 64
in_size = 64

# nhwc -> nchw
A = te.placeholder((batch, in_size, in_size, in_channel), name='A')

A_ch = te.compute(
    (batch, in_channel, in_size, in_size),
    lambda n, c, h, w: A[n, h, w, c],
    name='A_change')

s = te.create_schedule(A_ch.op)


block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

blockdim, threaddim = 32, 32
div = 4
n, c, h, w = s[A_ch].op.axis
hw = s[A_ch].fuse(h, w)
co, ci = s[A_ch].split(c, factor=threaddim)
hwo, hwi = s[A_ch].split(hw, factor=threaddim*div)
hwio, hwii = s[A_ch].split(hwi, factor=div)
hwiio, hwiii = s[A_ch].split(hwii, factor=4)
s[A_ch].bind(n, block_z)
s[A_ch].bind(co, block_y)
s[A_ch].bind(hwo, block_x)
s[A_ch].bind(ci, thread_y)
s[A_ch].bind(hwio, thread_x)
s[A_ch].reorder(n, co, hwo, ci, hwio, hwiio, hwiii)
s[A_ch].vectorize(hwiii)
print(tvm.lower(s, [A, A_ch], simple_mode=True))


func = tvm.build(s, [A, A_ch], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.uniform(size=(batch, in_size, in_size, in_channel)).astype(A.dtype)
a = tvm.nd.array(a_np, ctx)
a_ch = tvm.nd.array(np.zeros((batch, in_channel, in_size, in_size), dtype=A_ch.dtype), ctx)
func(a, a_ch)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
conv_time = evaluator(a, a_ch).mean * 1e3
tot_byte = batch * in_channel * in_size * in_size * 4 / 1024 / 1024 / 1024 # GB
print('Convolution: %f ms, Bandwidth: %f GB/s' % (conv_time, tot_byte / conv_time * 1000 * 2))


dev_module = func.imported_modules[0]
print(dev_module)
print("----GPU code----")
print(dev_module.get_source())

