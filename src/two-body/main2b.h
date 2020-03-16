#ifndef _main2b_h_
#define _main2b_h_
typedef uint64_t uint64;

struct input
{
  locate par0;
  locate par1;
  double mas0;
  double mas1;
  double cha0;
  double cha1;
  double elec;
  double radi;
  double time;
  uint64 numb;
};

struct locate
{
  double x;
  double y;
  double z;
};

struct output
{
  double time;
  locate par0;
  locate par1;
};

#endif
