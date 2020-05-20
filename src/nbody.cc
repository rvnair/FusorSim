#include <getopt.h>
#include <nbody.h>

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

int main(int argc, char** argv)
{
  intput in;
  parseInput(&in, argc, argv);

  locate* p = aligned_alloc(ALLOC_ALIGN, sizeof(locate) * in.part);
  locate* v = aligned_alloc(ALLOC_ALIGN, sizeof(locate) * in.part);
  double* m = aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);
  double* q = aligned_alloc(ALLOC_ALIGN, sizeof(double) * in.part);

  output* out = n_body_eval(p, v, m, q, in.elec, in.radi, in.time, in.numb, in.part);

  printf("The timestep is %lf\n", in.time);
  for(int i = 0; i < in.numb; i++)
  {
    printf("Timestep:\t%llu\n", out[i].step);
    for(uint64 j = 0; j < in.part)
    {
      printf("\tParticle %llu: %lf %lf %lf", j, out[i].part[j].x, out[i].part[j].y, out[i].part[j].z);
    }
  }
  return 0;
}
