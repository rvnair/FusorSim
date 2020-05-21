#include <sim.h>
#include <stdlib.h>

static void step(real_v* q, real_v* qdot, double* m, double* c, double dt, uint64 N, real_v* out, real_v* Fq1, void (*passed)(output), output prepped)
{

}

void n_body_eval(real_v* q, real_v* v, double* m, double* c, double QE, double r, double dt, uint64 n, uint64 N, void (*passed)(output))
{
    output prep;
    prep.numb = N;
    prep.part = nullptr;
    real_v* d_out = (real_v *) malloc(sizeof(real_v)*N*N);
    real_v* d_Fq1 = (real_v *) malloc(sizeof(real_v)*N); 

    for(uint64 i = 0; i < n; i++)
    {
        prep.step = i;
        step(q, v, m, c, dt, N, d_out, d_Fq1, passed, prep);
    }
}
