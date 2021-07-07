#ifndef FAKE_MUTATION
#define FAKE_MUTATION
#include "common.h"
#include "graph.h"
#include "operator.h"
#include "tensor.h"

namespace tpm {
class FakeMutation {
    std::vector<SubGraph *> candidates;

  public:
    FakeMutation() : candidates({}) {}
    ~FakeMutation() {
        for (auto sg : candidates)
            delete sg;
    }
    std::vector<SubGraph *> &run(SubGraph &graph) {
        candidates.emplace_back(new SubGraph(graph.getOperators()));
        auto &ops = graph.getOperators();
        if (ops.size() != 1)
            return candidates;
        if (ops[0]->getType() != Operator::Conv)
            return candidates;

        auto conv = dynamic_cast<ConvOp*>(ops[0]);
        auto args = conv->getArgs();
        if (args[ConvOp::SH] != 1 || args[ConvOp::SW] != 1 ||
            args[ConvOp::DH] != 2 || args[ConvOp::DW] != 2)
            return candidates;
        auto input = conv->getInputs()[0];
        auto weight = conv->getInputs()[1];
        auto trans = TransposeOp(input, 2, {0, 1, {-1, 2}, 3}, -2);
        auto convd1 =
            ConvOp(trans.getOutputs()[0], weight, args[ConvOp::PH] / 2,
                   args[ConvOp::PW] / 2, 1, 1, 1, 1);
        auto sb = new SubGraph({&trans, &convd1});
        candidates.emplace_back(sb);

        return candidates;
    }
};

} // end of namespace tpm
#endif // FAKE_MUTATION
