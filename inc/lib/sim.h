#ifndef _sim_h_
#define _sim_h_
#include <locate.h>
#define ALLOC_ALIGN 4096

output* n_body_eval(locate* p, locate* v, double* m, double* q, double QE, double r, double dt, uint64 n, uint64 N);

#endif
