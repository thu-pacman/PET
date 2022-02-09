#pragma once
#include <iostream>

namespace nnet {

// Dummy Expr
class ExprNode {
  public:
    std::string toReadable() { return ""; }
};
class VarNode {};
class TensorNode {};
class OperatorNode {};
class RangeOpNode {};
class SubscriptNode {};
class BinaryOpNode {};
class ConstantNode {};
class FuncNode {};
using Expr = std::shared_ptr<ExprNode>;
using Var = std::shared_ptr<VarNode>;
using Tensor = std::shared_ptr<TensorNode>;
using Operator = std::shared_ptr<OperatorNode>;
using RangeOp = std::shared_ptr<RangeOpNode>;
using Subscript = std::shared_ptr<SubscriptNode>;
using BinaryOp = std::shared_ptr<BinaryOpNode>;
using Constant = std::shared_ptr<ConstantNode>;
using Func = std::shared_ptr<FuncNode>;

} // namespace nnet
