#include "generator.h"
#include "graph.h"
#include "operator.h"

using namespace tpm;

int main() {
    auto g = Graph{};
    auto i0 = g.tensor({2, 4, 2, 2});
    auto i1 = g.tensor({2, 6, 2, 2});
    auto o0 = g.tensor({2, 4, 2, 2});
    auto o1 = g.tensor({2, 6, 2, 2});
    auto op0 = g.identity(i0, o0);
    auto op1 = g.identity(i1, o1);

    auto sg = SubGraph({op0, op1});
    for (auto tensor : sg.getTensors())
        tensor->dataMalloc();
    for (auto input : sg.getInputs())
        input->dataMalloc();
    for (auto op : sg.getOperators())
        op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    all_ops.emplace_back(new tpm::ConcatOp(1));
    all_ops.emplace_back(new tpm::SplitOp(1, std::vector<int>{2, 3}));
    tpm::Generator mutant{};
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(&sg, candidates, 2, all_ops);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates)
        candidate->print();

    return 0;
}