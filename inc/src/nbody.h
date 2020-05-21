#ifndef _nbody_h_
#define _nbody_h_
#include <locate.h>
#include <ifstream>

struct input
{
  char*  file;
  double elec;
  double radi;
  double time;
  uint64 numb;
  uint64 part;
};

#endif
