import onnx
from onnx import helper
from onnx import TensorProto


def mult_list(l):
    ret = 1
    for x in l:
        ret *= x
    return ret


def getProto():
    ret = {
        'Float32': TensorProto.FLOAT,
        'Int32': TensorProto.INT32,
        'Int64': TensorProto.INT64
    }
    return ret


def topo_sort(op_name, op_input, op_output):
    sorted_op, degree = list(), dict()
    for op in op_name:
        for tensor in op_input[op]:
            if not tensor in degree:
                degree[tensor] = 0

        for tensor in op_output[op]:
            if not tensor in degree:
                degree[tensor] = 0
            degree[tensor] += 1

    while True:
        have_zero_in = False
        for op in op_name:
            if op in sorted_op:
                continue
            max_deg = max([degree[v] for v in op_input[op]])
            if max_deg == 0:
                have_zero_in = True
                sorted_op.append(op)
                for tensor in op_output[op]:
                    degree[tensor] -= 1
        if not have_zero_in:
            break

    assert len(op_name) == len(
        sorted_op), 'The given computing graph is not a directed acyclic graph'

    return sorted_op


def unique_tensor(vec1, vec2):
    con = []
    for item in vec1:
        if item in vec2:
            con.append(item)

    ret1, ret2 = [], []
    for item in vec1:
        if not item in con:
            ret1.append(item)
    for item in vec2:
        if not item in con:
            ret2.append(item)

    return ret1, ret2


def export_onnx(path, tensor_name, tensor_dtype, tensor_dim, initializer_name,
                op_name, op_input, op_output, op_attr, tensor_value):
    
    print("========================================================================")
    print(f"tensor_name:")
    print(f"length: {len(tensor_name)}")
    print(tensor_name)
    print("========================================================================")
    print(f"tensor_dtype:")
    print(f"length: {len(tensor_dtype)}")
    print(tensor_dtype)
    print("========================================================================")
    print(f"tensor_dim:")
    print(f"length: {len(tensor_dim)}")
    print(tensor_dim)
    print("========================================================================")
    print(f"initializer_name:")
    print(f"length: {len(initializer_name)}")
    print(initializer_name)
    print("========================================================================")
    print(f"op_name:")
    print(f"length: {len(op_name)}")
    print(op_name)
    print("========================================================================")
    print(f"op_input:")
    print(f"length: {len(op_input)}")
    print(op_input)
    print("========================================================================")
    print(f"op_output:")
    print(f"length: {len(op_output)}")
    print(op_output)
    print("========================================================================")
    print(f"op_attr:")
    print(f"length: {len(op_attr)}")
    print(op_attr)
    print("========================================================================")
    print(f"tensor_value:")
    print(f"length: {len(tensor_value)}")
    print(tensor_value)
    print("========================================================================")

    op_name = topo_sort(op_name, op_input, op_output)

    nodes, net_input, net_output = list(), list(), list()
    tensors = dict()

    for i, name in enumerate(op_name):
        attr = op_attr[name]
        for key in attr.keys():
            if attr[key][0] == '[' or attr[key].isdigit():
                # convert string format to list in Python
                attr[key] = eval(attr[key])

        op_type = name.split('_')[0]
        if op_type == 'Reshape':
            shape_name = name + '_shape'
            op_input[name].append(shape_name)

            shape = tensor_value[shape_name]
            print(f"Reshape shape: {shape}")
            tensor_dim[shape_name] = [len(shape)]
            tensor_dtype[shape_name] = 'Int64'
            tensor = helper.make_tensor_value_info(
                shape_name,
                TensorProto.INT64,
                [len(shape)]
            )
            tensors[shape_name] = tensor
            initializer_name.append(shape_name)

        node_act = None

        if op_type == 'Conv' and 'act' in attr:
            ins = op_output[name][0] + '_' + attr['act']
            oup = op_output[name][0]
            node = helper.make_node(
                attr['act'],
                [ins],
                op_output[name]
            )
            node_act = node
            op_output[name] = [ins]
            net_input.append(ins)
            net_output.append(oup)
            attr.pop('act')

            tensor_name.append(ins)
            tensor_dtype[ins] = tensor_dtype[oup]
            tensor_dim[ins] = tensor_dim[oup]

        node = helper.make_node(
            op_type,
            op_input[name],
            op_output[name],
            **attr
        )
        nodes.append(node)
        if node_act != None:
            nodes.append(node_act)

        for ten in op_input[name]:
            if not ten in net_input:
                net_input.append(ten)

        for ten in op_output[name]:
            if not ten in net_output:
                net_output.append(ten)
            if ten in initializer_name:
                initializer_name.remove(ten)

        print(
            f'i: {i}, op: {name}, input: {op_input[name]}, output: {op_output[name]}, attr: {attr}')

    net_input, net_output = unique_tensor(net_input, net_output)

    for name in tensor_name:
        dtype = tensor_dtype[name]
        tensor = helper.make_tensor_value_info(
            name,
            getProto()[dtype],
            tensor_dim[name]
        )
        tensors[name] = tensor

    initializers = list()
    for name in initializer_name:
        dtype = tensor_dtype[name]
        init = helper.make_tensor(
            name,
            getProto()[dtype],
            tensor_dim[name],
            [0] * mult_list(tensor_dim[name]
                            ) if not name in tensor_value else tensor_value[name]
        )
        initializers.append(init)

    print(f'net_input {net_input}')
    print(f'net_output {net_output}')

    graph_def = helper.make_graph(
        nodes,
        'pet-optimization',
        [tensors[name] for name in net_input],
        [tensors[name] for name in net_output],
        initializers
    )

    model_def = helper.make_model(graph_def, producer_name='PET')

    print(onnx.checker.check_model(model_def))
    onnx.save(model_def, path)
    