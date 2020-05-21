#ifndef _sim_h_
#define _sim_h_
#include <locate.h>
constexpr int ALLOC_ALIGN = 4096;

void n_body_eval(real_v* p, real_v* v, double* m, double* q, double QE, double r, double dt, uint64 n, uint64 N, void(*passed)(output));

#endif
