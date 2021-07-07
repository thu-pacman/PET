import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import sys
import os
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if len(sys.argv) != 8:
    print("Usage: python %s <pd-file> n c d h w" % (sys.argv[0]))
    exit(-1)

PB_PATH = sys.argv[1]

n = int(sys.argv[2])
c = int(sys.argv[3])
d = int(sys.argv[4])
h = int(sys.argv[5])
w = int(sys.argv[6])
xla = eval(sys.argv[7])

print("params", PB_PATH, n, c, d, h, w, xla)

# INPUT_NAME = 'input.1:0'
# OUTPUT_NAME = '191:0'

if h==224:
    INPUT_NAME = 'input.1:0'
    OUTPUT_NAME = '191:0'
elif h==330:
    INPUT_NAME = 'x.1:0'
    OUTPUT_NAME = '924:0'
elif h==14:
    INPUT_NAME = 'input.1:0'
    OUTPUT_NAME = '18:0'
elif h==512:
    INPUT_NAME = 'input.1:0'
    OUTPUT_NAME = '444:0'
elif h==112:
    INPUT_NAME = '0:0'
    OUTPUT_NAME = '190:0'
else:
    assert(False)

# x = np.random.randn(n * c * h * w)
x = np.random.randn(n, c , d , h , w)
# y = tf.constant(list(x), shape=(n, c, h, w), dtype='float32')

# XLA
tf.config.optimizer.set_jit(xla)

with tf.gfile.FastGFile(PB_PATH, 'rb') as fin:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fin.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

input_tensor = graph.get_tensor_by_name(INPUT_NAME)
output_tensor = graph.get_tensor_by_name(OUTPUT_NAME)


with tf.Session(graph=graph) as sess:
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()
    sess.run(output_tensor, feed_dict={
             input_tensor: x}, options=run_options, run_metadata=run_metadata)

    print("Start timing")
    t0 = time()
    n_iter=100
    for _ in range(n_iter):
        sess.run(output_tensor, feed_dict={
                 input_tensor: x}, options=run_options, run_metadata=run_metadata)
    print('Time %d iter'%n_iter, time() - t0)
    print('Averager', (time() - t0)/n_iter*1000, 'ms')



# tl = timeline.Timeline(run_metadata.step_stats)
# ctf = tl.generate_chrome_trace_format()

# with open("timeline.js", 'w') as fout:
#     fout.write(ctf)
