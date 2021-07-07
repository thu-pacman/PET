import tvm
from tvm import te
import topi
from tvm import autotvm
import numpy as np
import logging
import sys

n, c, h, w = 16, 256, 28, 28
f, r, s = 512, 3, 3
stride = (1, 1)
padding = (1, 1)
dilate = (1, 1)
groups = 4
dtype = 'float32'
c_perg = int(c/groups)

input_shape = (n, c, h, w)
weight_shape = (f, c_perg, r, s)
ap = tvm.te.placeholder(shape=input_shape, dtype=dtype, name="A")
wp = tvm.te.placeholder(shape=weight_shape, dtype=dtype, name="W")

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

task = autotvm.task.create('group_conv2d_nchw.cuda', args=(ap, wp, stride, padding, dilate, groups, dtype), target='cuda')

measure_option = autotvm.measure_option(
    builder = autotvm.LocalBuilder(),
    runner = autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
)
tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')

tuner.tune(n_trial = 2000,
            measure_option = measure_option,
            callbacks = [autotvm.callback.log_to_file('conv-grouped.log')]
)

dispatch_context = autotvm.apply_history_best('conv-grouped.log')
best_config = dispatch_context.query(task.target, task.workload)
print(best_config)

with autotvm.apply_history_best('conv-grouped.log') as best:
    with tvm.target.create('cuda'):
        outs = topi.cuda.group_conv2d_nchw(ap, wp, stride, padding, dilate, groups, dtype)
        s = topi.cuda.schedule_group_conv2d_nchw([outs])
        func = tvm.build(s, [ap, wp])

ctx = tvm.gpu(0)
a_np = np.random.uniform(size=input_shape).astype(dtype)
w_np = np.random.uniform(size=weight_shape).astype(dtype)
a = tvm.nd.array(a_np, ctx)
w = tvm.nd.array(w_np, ctx)

evaluator = func.time_evaluator(func.entry_name, ctx, number = 1, repeat = 10)
timer = evaluator(a, w).mean*1e3
print("time: %.2fms" % (timer))

# dev_module = func.imported_modules[0]
# print(dev_module)
# print("-------GPU code-------")
# print(dev_module.get_source())
