import sys

import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
import logging

if len(sys.argv) != 2:
    print("Usage: %s <onnx-file>" % sys.argv[0])
    exit(1)

onnx_model = onnx.load(sys.argv[1])


input_name = "input.1"
x = np.random.randn(10, 3, 224, 224)

# # mybert
# input_name = "data"
# x = np.random.randn(64, 1024)

######################################################################
# Compile the model with relay
# ---------------------------------------------
target = "cuda  -libs=cudnn,cublas"

shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# logging.getLogger("compile_engine").setLevel(logging.INFO)
# logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))

with tvm.transform.PassContext(opt_level=3):
    # Don't use GraphExecutor, which is for debugging only
    lib = relay.build(mod, target=target, params=params)

ctx = tvm.gpu(0)
dtype = "float32"
data_tvm = tvm.nd.array(x.astype(dtype), ctx=ctx)
module = tvm.contrib.graph_runtime.GraphModule(lib["default"](ctx))
module.set_input(input_name, data_tvm)

print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=50)
prof_res = np.array(ftimer().results) * 1e3
print("Time cost is: ", np.mean(prof_res), "ms", " stddev = ", np.std(prof_res), "ms")

