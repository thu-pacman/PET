#include "simulator.h"
#include "operator.h"
#include <cstdio>

void Simulator::init_ops() {
	int bs, n, c, h, w, f, r, s, g;
	
	// TODO: get args from a config-file

}

void Simulator::measure_all_ops(int rounds) {
	for (auto op : ops_vec) {
		op->performance_measuring(this, rounds);
	}
}
