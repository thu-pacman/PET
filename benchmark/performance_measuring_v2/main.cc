// An example

#include <iostream>
#include <vector>
#include "simulator.h"
#include "operator.h"

int main(int argc, char* argv[]) {
	std::cout << "Test performence of operators\n";

	std::vector<OpConfig> configVector;
	
	OpConfig conv2dConfig;
	conv2dConfig.opType = OpType::CONV2D;
	conv2dConfig.args   = {
		/* Group_count = */ 1,
		/* Batch_size  = */ 64,
		/* In_channels_per_grp =  */ 64,
		/* In_h        = */ 28,
		/* In_w        = */ 28,
		/* Out_channels= */ 128,
		/* Kn_h        = */ 3,
		/* Kn_w        = */ 3,
		/* Pad_h       = */ 1,
		/* Pad_w       = */ 1,
		/* Stride_h    = */ 1,
		/* Stride_w    = */ 1,
		/* Dila_h      = */ 1,
		/* Dila_w      = */ 1
	};

	OpConfig matMulConfig;
	matMulConfig.opType = OpType::MATMUL;
	matMulConfig.args   = {
		/* M = */ 128 * 3 * 3,
		/* N = */ 64 * 28 * 28,
		/* K = */ 64,
		/* transa = */ 0,
		/* transb = */ 0,
		/* tenser_op = */ 0,
		/* algo   = */ -1
	};

	configVector.push_back(conv2dConfig);
	configVector.push_back(matMulConfig);

	Simulator* simu = new Simulator;
	simu->initOp(configVector);
	PfMap pfMap = simu->measureAllOp(10);
	simu->freeOp();
	delete simu;

	std::cout << "End of test\n";
	return 0;
}
