#ifndef _sim_h_
#define _sim_h_
#include <real_v.h>
constexpr int ALLOC_ALIGN = 4096;

struct output
{
  uint64  step; // The current iteration
  uint64  numb; // The current number of particles in the simulation
  real_v* part; // the positions of each particle
};

void n_body_eval(real_v* p, real_v* v, double* m, double* q, double QE, double r, double dt, uint64 n, uint64 N, void(*passed)(output));

#endif
