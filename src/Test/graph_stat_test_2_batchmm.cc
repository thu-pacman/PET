#include "generator.h"
#include "graph.h"
#include "operator.h"
#include "tensor.h"
#include <iostream>

const int m = 8, n = 8, k = 4;

using namespace tpm;

int main() {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, m, k});
    auto w0 = g->tensor({1, k, n});
    auto w1 = g->tensor({1, k, n});
    auto i1 = g->tensor({1, m, n});
    auto i2 = g->tensor({1, m, n});

    auto w2 = g->tensor({1, k, n + 1});
    auto i3 = g->tensor({1, m, n + 1});

    auto op0 = g->matmul(i0, w0, i1);
    auto op1 = g->matmul(i0, w1, i2);
    auto op2 = g->matmul(i0, w2, i3);

    auto sg = SubGraph({op0, op1});
    auto gen = Generator();
    std::cout << gen.statGraph(&sg) << std::endl;

    sg = SubGraph({op0, op2});
    std::cout << gen.statGraph(&sg) << std::endl;

    return 0;
}