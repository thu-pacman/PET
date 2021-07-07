#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <map>
#include <string>
#include <vector>


class Operator;
class PerformanceInfo;

class Simulator {

public:
  Simulator() { init_ops(); }

  void measure_all_ops(int rounds);
  // void print_res_info();

private:
  void init_ops();

  std::vector<Operator *> ops_vec;
  std::map<std::string, PerformanceInfo *> pf_info_map;
};

#endif
