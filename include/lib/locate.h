#ifndef _locate_h_
#define _locate_h_
#include <cstdint>
typedef uint64_t uint64;

struct real_v
{
  double vals[3];
};

struct output
{
  uint64  step;
  real_v* part;
};
#endif
