# FusorSim

To build:
``` bash
mkdir build
cd build
cmake ..
make -j
```

All binaries will then be in the build folder.


## libsim
  This library contains all the simulation functions.
  Each function takes in a function pointer that recieves the particular positions of all the particles.
  Look at `inc/lib/sim.h` and find the ouput struct for more information.

## libvis
  This library will contain visualization functions.
