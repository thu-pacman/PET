#include "operator.h"
#include "simulator.h"

int main(int argc, char* argv[]) {
	Simulator* simu = new Simulator;

	simu->measure_all_ops(1);

	delete simu;
	
	return 0;
}
