import numpy as np
import tvm
from tvm import te

def gen_simple_op(input_tensors, input_dtypes, output_tensor, f, func_name, input_names, output_name):
    assert len(input_tensors) == len(input_dtypes)
    assert len(input_tensors) == len(input_names)

    I = []
    for i, (shape, dtype) in enumerate(zip(input_tensors, input_dtypes)):
        I.append(te.placeholder(shape, name='I%d' % i, dtype=dtype))
    A_ch = te.compute(output_tensor, eval(f, {'I': I, 'tvm': tvm}), name='A_change')
    tensors = I + [A_ch]

    s = te.create_schedule(A_ch.op)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    block_dim, thread_dim = 640, 1024
    axis = s[A_ch].op.axis
    all_fuse = axis[0]
    for _axis in axis[1:]:
        all_fuse = s[A_ch].fuse(all_fuse, _axis)
    block_o, block_i = s[A_ch].split(all_fuse, nparts=block_dim)
    thread_o, thread_i = s[A_ch].split(block_i, nparts=thread_dim)
    s[A_ch].bind(block_o, block_x)
    s[A_ch].bind(thread_o, thread_x)
    func = tvm.build(s, tensors, 'cuda', name=func_name)
    dev_module = func.imported_modules[0]

    output_dims = 1
    for d in output_tensor:
        output_dims *= d

    func_code = dev_module.get_source()
    invoke_code = "%s_kernel0<<<%d, %d>>>(%s, %s);" % (
            func_name, min(block_dim, output_dims), thread_dim, output_name, ", ".join(input_names))

    func_code = "// %s -> %s:\n// %s\n%s" % (input_tensors, output_tensor, f, func_code)

    ctx = tvm.cuda(0)
    input_a = []
    for i, (shape, dtype) in enumerate(zip(input_tensors, input_dtypes)):
        a_np = np.random.uniform(size=shape).astype(dtype)
        input_a.append(tvm.nd.array(a_np, ctx))
    a_ch = tvm.nd.array(np.zeros(output_tensor, dtype=A_ch.dtype), ctx)
    func(*input_a, a_ch)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    conv_time = evaluator(*input_a, a_ch).mean * 1e3

    return func_code, invoke_code, conv_time # ms

