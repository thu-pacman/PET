from .core import *
import analyze_onnx

def solve():
    tensors, ops = analyze_onnx.getTensorOp()
    
    graph = core.PyGraph()
    pytensors = dict()
    for name, tensor in tensors:
        pytensors[name] = graph.tensor(tensor.guid, tensor.dims)
    
    for name, op in ops:
        if op.OP_TYPE == 'Conv':
            graph.conv(
                op.guid,
                pytensors[op.input[0].name],
                pytensors[op.input[1].name],
                pytensors[op.output[0].name],
                op.args['pads'][2],
                op.args['pads'][3],
                op.args['strides'][0],
                op.args['strides'][1],
                op.args['dilations'][0],
                op.args['dilations'][1]
            )
        if op.OP_TYPE == 'Mul':
            graph.matmul(
                op.guid,
                pytensors[op.input[0].name],
                pytensors[op.input[1].name],
                pytensors[op.input[2].name]
            )
        if op.OP_TYPE == 'Concat':
            graph.concat(
                op.guid,
                [pytensors[_.name] for _ in op.input],
                pytensors[op.output[0].name],
                op.args['axis'][0] # un
            )
        if op.OP_TYPE == 'Relu':
            graph.relu(
                op.guid,
                pytensors[op.input[0].name],
                pytensors[op.output[0].name]
            )
        

if __name__ == '__main__':
    solve()
