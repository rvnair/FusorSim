#include <getopt.h>
#include "nbody.h"
#include "gpu/repeat_iterator.h"

void parseInput(input* in, int argc, char** argv)
{
  int c;
  while ((c = getopt(argc, argv, "f:Q:R:s:n:N")) != -1)
  {
    switch(c)
    {
      case 'f':
        in->file  = optarg;
        break;
      case 'Q':
        in->elec  = stod(optarg);
        break;
      case 'R':
        in->radi  = stod(optarg);
        break;
      case 's':
        in->time  = stod(optarg);
        break;
      case 'n':
        in->numb  = stoull(optarg);
        break;
      case 'N':
        in->part  = stoull(optarg);
        break;
      default:
        abort();
    }
  }
}

void printer(output out)
{
  printf("Timestep:\t%llu\n", step);
  for(uint64 j = 0; j < out.numb)
  {
    printf("\tParticle %llu: %lf %lf %lf", j, out[i].part[j].x, out[i].part[j].y, out[i].part[j].z);
  }
}

int main(int argc, char** argv)
{
  intput in;
  parseInput(&in, argc, argv);

  real_v* q = aligned_alloc(ALLOC_ALIGN, sizeof(locate) * in.part);
  real_v* v = aligned_alloc(ALLOC_ALIGN, sizeof(locate) * in.part);
  double* m = aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);
  double* c = aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);
  // The input file stream should have particles in q0 q1 q2 v0 v1 v2 m c
  std::ifstream inf(in.file);
  for(int i = 0; i < N; i++)
  {
    for(int j = 0; j < REAL_V_DIM; j++) inf >> q[i].vals[j];
    for(int j = 0; j < REAL_V_DIM; j++) inf >> v[i].vals[j];
    inf >> m[i];
    inf >> c[i];
  }
  printf("The timestep is %lf\n", in.time);
  n_body_eval(q, v, m, c, in.elec, in.radi, in.time, in.numb, in.part, printer);
  return 0;
}
