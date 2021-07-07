'''
Example usage:

On host machine:
python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 &

On Arm marchine:
(Don't forget numactl)
python3 -m tvm.exec.rpc_server --tracker=x.x.x.x:9190 --key=hihope &

On host machine:
python3 -m tvm.exec.query_rpc_tracker --host=127.0.0.1 --port=9190
time python3 tune_tvm_conv2d.py TUNE 64 128 128 224 224 3
'''

import logging
import json
import time
import sys
import os

import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
from tvm import topi

logging.getLogger("compile_engine").setLevel(logging.INFO)
logging.getLogger("compile_engine").addHandler(logging.StreamHandler(sys.stdout))

if len(sys.argv) != 4:
    print("Usage: %s <TUNE|EVAL> l r" % sys.argv[0])
    exit(-1)
if sys.argv[1] == "TUNE":
    TUNE = True
elif sys.argv[1] == "EVAL":
    TUNE = False
else:
    assert False
idl = int(sys.argv[2])
idr = int(sys.argv[3])
# data_layout, kernel_layout = sys.argv[2:4]
# assert data_layout == "NCHW" or data_layout == "NHWC"
# assert kernel_layout == "OIHW" or kernel_layout == "OHWI"
data_layout = "NCHW"
kernel_layout = "OIHW"
# N, I, O, H, W, K = map(int, sys.argv[4:])
# strides = (1, 1)
# padding = ((K - 1) // 2, (K - 1) // 2)

dtype = "float32"
#target = "opencl -device=bifrost -model=g71"
# target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
target = "cuda"
# arget = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon -libs=nnpack"
# target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
# target_host = "cuda"

# log_file = "topi_conv2d_%s_%s_%d_%d_%d_%d_%d_%d_%s.log"%(data_layout, kernel_layout, N, I, O, H, W, K, target.split()[0])
log_file = "topi_bmm_1.log"+str(idl)+'-'+str(idr)


def get_workload_conv2d(N, I, O, H, W, K, strides, padding, dilation, groups, dtype):
    shape = {
        "data": (N, I, H, W) if data_layout == "NCHW" else (N, H, W, I),
        "weight": (O, I, *K) if kernel_layout == "OIHW" else (O, K, K, I),
    }
    data = relay.var("data", shape=shape["data"], dtype=dtype)
    print('shape=', shape["weight"])
    weight = relay.var("weight", shape=shape["weight"], dtype=dtype)
    out = relay.nn.conv2d(data, weight,
                          channels=O, kernel_size=K,
                          strides=strides, padding=padding, dilation=dilation, data_layout=data_layout, groups=groups, kernel_layout=kernel_layout)
    params = {}
    for key in ("weight",):  # "data" is NOT a parameter, because parameters are folded as constants
        init_val = np.random.uniform(-1, 1, size=shape[key]).astype(dtype)
        params[key] = tvm.nd.array(init_val)
    return relay.Function([data, weight], out), params, shape["data"]


def get_workload_bmm(M, N, K, B, dtype):
    shape = {
        "A": (B, M, K),
        "B": (B, N, K),
    }
    A = relay.var("A", shape=shape['A'], dtype=dtype)
    B = relay.var("B", shape=shape['B'], dtype=dtype)
    out = tvm.relay.nn.batch_matmul(A, B)
    params = {}
    for key in ("A", "B"):  # "data" is NOT a parameter, because parameters are folded as constants
        init_val = np.random.uniform(-1, 1, size=shape[key]).astype(dtype)
        params[key] = tvm.nd.array(init_val)
    return relay.Function([A, B], out), params, shape

def matmul():
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((N, K), name="B", dtype=dtype)
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name="matmul")
    return [A, B, C]

def tune_kernels(tasks, measure_option, tuner, early_stopping, log_filename, n_trial):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        print("%s FLOPs / task" % task.flop)
        print(task.config_space)

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        print("trials=", n_trial)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(
                               n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_file)])

        time.sleep(5)  # Bug?


# Log the chosen algorithm
logging.getLogger("compile_engine").setLevel(logging.INFO)
logging.getLogger("compile_engine").addHandler(
    logging.StreamHandler(sys.stdout))


tuning_opt = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 1024,  # 1024,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            timeout=20, number=5, repeat=1, min_repeat_ms=300),

        # "hihope", "127.0.0.1", 9190,
        # number=5, repeat=1, min_repeat_ms=1000, timeout=10000)
        # Ansor exp setting:
        # runner=autotvm.RPCRunner(
        #     "hihope", "127.0.0.1", 9190,
        #     number=5, repeat=1, min_repeat_ms=1000, timeout=10000)
    )
}

# https://docs.google.com/spreadsheets/d/1it9z3boqPLeKhjUmVgAX3WWuqJ5aBpUbg5xhOlrWVYc/edit#gid=1102136610
input_tasks = [  # Input, Kernel, p, s, d
    [512, 768, 768, 1],
    [512, 768, 768, 3],
]

task_id = 0

if TUNE:  # reversed tuning for faster results
    ran = range(idr-1, idl-1, -1)
else:
    ran = range(idl, idr)

for task_id in ran:
    print('task_id=', task_id)
    atask = input_tasks[task_id]
    M, N, K, B = atask
    print(M, N, K, B)
    op, params, data_shape = get_workload_bmm(M, N, K, B, dtype)
    op = tvm.IRModule.from_expr(op)

    log_file = "relay_bmm_"+str(atask)+'.log'
    log_file = log_file.replace(' ', '')
    tuning_opt['log_filename'] = log_file
    print(log_file)
# sys.exit()

# def get_workload(N, I, O, H, W, K, strides, padding, dilation, group, dtype):

# Use Arm Compute Library
#from tvm.relay.op.contrib.arm_compute_lib import partition_for_arm_compute_lib
#op = partition_for_arm_compute_lib(op, params)

    if TUNE:
        tasks = autotvm.task.extract_from_program(op, target='cuda',
                    params=params, ops=(relay.op.get("nn.batch_matmul"), ))
        print(tasks)
        if len(tasks) > 1:
            tasks = tasks[:1]
        print("Drop extra tasks")
        print(tasks)
        print("Tuning...")
        tune_kernels(tasks, **tuning_opt)

        # print()
        # with autotvm.apply_history_best(log_file) as best:
        #     for key in best.best_by_targetkey:
        #         print(best.best_by_targetkey[key])
        #         exit(0)
    else:
        print("# BEST TUNED")
        try:
            with open(log_file) as f:
                min_cost = 9999999999
                for l in f.readlines():
                    measured = autotvm.record.decode(l)[1]._asdict()
                    if measured['error_no'] == 0:
                        min_cost = min(min_cost, measured['costs'][0])
                print('BestTime=', min_cost*1000)
        except FileNotFoundError:
            print('Not Found')
        # with autotvm.apply_history_best(log_file) as best:
        # print("Compile...")
        # with tvm.transform.PassContext(opt_level=3):
        #     # with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        #     lib = relay.build(op, target=target,
        #                       target_host=target_host, params=params)

        # # export library
        # tmp = tvm.contrib.util.tempdir()
        # filename = "net.tar"
        # lib.export_library(tmp.relpath(filename))

        # # upload module to device
        # print("Upload...")
        # remote = autotvm.measure.request_remote(
        #     "hihope", '127.0.0.1', 9190, timeout=10000)
        # remote.upload(tmp.relpath(filename))
        # rlib = remote.load_module(filename)

        # # upload parameters to device
        # ctx = remote.context(target, 0)
        # data_tvm = tvm.nd.array(
        #     (np.random.uniform(size=data_shape)).astype(dtype), ctx=ctx)
        # module = tvm.contrib.graph_runtime.GraphModule(
        #     rlib["default"](ctx))
        # module.set_input("data", data_tvm)

        # # evaluate
        # print("Evaluate inference time cost...")
        # ftimer = module.module.time_evaluator(
        #     "run", ctx, number=1, repeat=50)
        # prof_res = np.array(ftimer().results) * 1e3
        # print("Time cost is: ", np.mean(prof_res), "ms",
        #       " stddev = ", np.std(prof_res), "ms")
