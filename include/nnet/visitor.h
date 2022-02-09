#pragma once
#include "expr.h"
#include <iostream>
#include <map>
#include <vector>

namespace nnet {

class AsTVMVisitor {
  private:
    int nStage = 0, curStage = -1;
    std::unordered_map<std::string, int> offset;
    std::vector<std::string> inputs;
    std::string output;
    std::vector<std::string> pythonVars;
    std::vector<std::vector<int>> inputShapes;
    std::vector<int> outputShape;
    std::string stmts;

  public:
    std::string getStmts() const { return ""; };

    const std::vector<std::string> &getInputs() const { return inputs; }
    const std::string &getOutput() const { return output; }

    const std::vector<std::vector<int>> &getInputShapes() const {
        return inputShapes;
    }
    const std::vector<int> &getOutputShape() const { return outputShape; }

    std::string visit_(const Constant &c) { return ""; };
    std::string visit_(const BinaryOp &c) { return ""; };
    std::string visit_(const Func &c) { return ""; };
    std::string visit_(const RangeOp &c) { return ""; };
    std::string visit_(const Subscript &c) { return ""; };
    std::string visit_(const Var &c) { return ""; };
    std::string visit_(const Tensor &c) { return ""; };
};

} // end of namespace nnet