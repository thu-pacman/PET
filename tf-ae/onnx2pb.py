import onnx
from onnx_tf.backend import prepare
import sys

if len(sys.argv) != 3:
    print("Usage: python %s <onnx-file> <path-to-where-pb-file-saved>\n" %
          (sys.argv[0]))
    exit(1)

print("exporting {} to {}".format(sys.argv[1], sys.argv[2]))

onnx_model = onnx.load(sys.argv[1])
tf_rep = prepare(onnx_model)
tf_rep.export_graph(sys.argv[2])
