#include <getopt.h>
#include "two_body.h"

void parseInput(input* in, int argc, char** argv)
{
  int c;
  while ((c = getopt(argc, argv, "x:y:z:X:Y:Z:q:Q:m:M:P:R:s:n:")) != -1)
  {
    switch(c)
    {
      case 'x':
        in->par0.x = stod(optarg);
        break;
      case 'y':
        in->par0.y = stod(optarg);
        break;
      case 'z':
        in->par0.z = stod(optarg);
        break;
      case 'X':
        in->par1.x = stod(optarg);
        break;
      case 'Y':
        in->par1.y = stod(optarg);
        break;
      case 'Z':
        in->par1.z = stod(optarg);
        break;
      case 'q':
        in->cha0   = stod(optarg);
      case 'Q':
        in->cha1   = stod(optarg);
      case 'm':
        in->mas0   = stod(optarg);
      case 'M':
        in->mas1   = stod(optarg);
      case 'P':
        in->elec   = stod(optarg);
        break;
      case 'R':
        in->radi   = stod(optarg);
        break;
      case 's':
        in->time   = stod(optarg);
        break;
      case 'n':
        in->numb   = stoull(optarg);
      default:
        abort();
    }
  }
}

int main(int argc, char** argv)
{
  intput in;
  parseInput(&in, argc, argv);
  output* out = two_body_eval(in.par0, in.mas0, in.cha0, in.par1, in.mas1, in.cha1, in.elec, in.radi, in.time, in.numb);

  for(int i = 0; i < in.numb; i++)
  {
    printf("%lf:\t%lf\t%lf\t%lf,\t%lf\t%lf\t%lf\n", out[i].time, out[i].par0.x, out[i].par0.y, out[i].par0.z, out[i].par1.x, out[i].par1.y, out[i].par1.z);
  }
  return 0;
}
