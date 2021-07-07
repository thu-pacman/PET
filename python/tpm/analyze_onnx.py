import onnx
import onnx.checker
import onnx.shape_inference

class Tensor:
    def __init__(self, name, guid, dims):
        self.name = name # str
        self.guid = guid # int
        self.dims = dims # tuple(int)

    def __str__(self):
        return 'name: %s, dims: %a' % (self.name, self.dims)

    def __repr__(self):
        return self.__str__()

class Op:
    def __init__(self, name, guid, typ, args = {}):
        self.name = name
        self.guid = guid
        self.type = typ
        self.args = args
        self.input = list()
        self.output = list()

class ConvOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Conv'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        for arg in attribute:
            if arg.name == 'group':
                self.args['group'] = arg.i
            else:
                self.args[arg.name] = arg.ints
                
class ConstantOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Constant'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        pass # TODO

class GatherOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Gather'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        pass # TODO     
    
class MulOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Mul'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        pass
    
class ConcatOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Concat'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        for arg in attribute:
            if arg.name == 'axis':
                self.args['axis'] = arg.i
    
class ReluOp(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = 'Relu'
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        pass    

"""
template of operations class
class Op(Op):
    def __init__(self, name, guid, args = {}):
        self.OP_TYPE = ''
        super().__init__(name, guid, self.OP_TYPE, args)

    def updateArgs(self, attribute):
        print(attribute)
"""

def getOpDict():
    opdict = dict()
    opdict['Conv'] = ConvOp
    opdict['Mul'] = MulOp
    opdict['Concat'] = ConcatOp
    opdict['Relu'] = ReluOp
    # opdict['Constant'] = ConstantOp
    # opdict['Gather'] = GatherOp
    
    return opdict

def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        existing_info = {vi.name: vi for vi in graph.value_info}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = existing_info.get(init.name)
            if vi is None:
                vi = graph.value_info.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)


    return add_const_value_infos_to_graph(model.graph)
    
def getTensorOp(net='inception.onnx'):
    tensors, ops = dict(), dict() # (key, value) = (name, class)
    model = onnx.load(net)
    now_guid = 0

    # Tensor_input
    for input in model.graph.input:
        if input.name not in tensors:
            dims = [d.dim_value for d in input.type.tensor_type.shape.dim]
            tensors[input.name] = Tensor(input.name, now_guid, tuple(dims))
            now_guid += 1
    
    # Tensor_weight
    for weight in model.graph.initializer:
        if weight.name not in tensors:
            tensors[weight.name] = Tensor(weight.name, now_guid, tuple(weight.dims))
            now_guid += 1
    
    # Tensor_inference
    add_value_info_for_constants(model)
    infered_model = onnx.shape_inference.infer_shapes(model)
    for v in infered_model.graph.value_info:
        if v.name not in tensors:
            dims = [d.dim_value for d in v.type.tensor_type.shape.dim]
            tensors[v.name] = Tensor(v.name, now_guid, tuple(dims))
            now_guid += 1
    
    # Tensor_output
    for output in model.graph.output:
        if output.name not in tensors:
            dims = [d.dim_value for d in output.type.tensor_type.shape.dim]
            tensors[output.name] = Tensor(output.name, now_guid, tuple(dims))
            now_guid += 1
    
    Oper_dict = getOpDict()

    # Op
    for node in model.graph.node:
        if node.name not in ops and node.op_type in Oper_dict.keys():
            ops[node.name] = Oper_dict[node.op_type](node.name, now_guid)
            for input in node.input:
                ops[node.name].input.append(tensors[input])
            for output in node.output:
                ops[node.name].output.append(tensors[output])
            ops[node.name].updateArgs(node.attribute)
            
    return tensors, ops

# TODO: delete
if __name__ == '__main__':
    getTensorOp()

