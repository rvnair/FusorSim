#include "nbody.h"
#include <sstream>
#include <iostream>

void parseInput(input* in, int argc, char** argv)
{
  int c;
  while ((c = getopt(argc, argv, "f:Q:R:s:n:N")) != -1)
  {
    std::istringstream stream(optarg);
    switch(c)
    {
      case 'f':
        in->file  = optarg;
        break;
      case 'Q':
        stream >> in->elec;
        break;
      case 'R':
        stream >> in->radi;
        break;
      case 's':
        stream >> in->time;
        break;
      case 'n':
        stream >> in->numb;
        break;
      case 'N':
        stream >> in->part;
        break;
      default:
        abort();
    }
  }
}

void printer(output out)
{
  std::cout << "Timestep:\t" << out.step << std::endl;
  for(uint64 j = 0; j < out.numb; j++)
  {
    std::cout << "Particle " << j << ":";
    for(int i = 0; i < REAL_V_DIM; i++) std::cout << "\t" << out.part[j].vals[i];
    std::cout << std::endl;
  }
}

int main(int argc, char** argv)
{
  input in;
  parseInput(&in, argc, argv);

  real_v* q = (real_v*) aligned_alloc(ALLOC_ALIGN, sizeof(real_v) * in.part);
  real_v* v = (real_v*) aligned_alloc(ALLOC_ALIGN, sizeof(real_v) * in.part);
  double* m = (double*) aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);
  double* c = (double*) aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);
  // The input file stream should have particles in q0 q1 q2 v0 v1 v2 m c
  std::ifstream inf(in.file);
  for(int i = 0; i < in.part; i++)
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
