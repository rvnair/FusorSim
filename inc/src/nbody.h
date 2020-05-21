#ifndef _nbody_h_
#define _nbody_h_
#include <fstream>
#include <getopt.h>
#include "real_v.h"
#include "sim.h"

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
