## Short build and run instructions.

### Clone/Checkout repository

```
git clone https://github.com/efocht/hpcg-ve.git
cd hpcg-ve
```

### Activate NCC and NEC MPI

For example:
```
export NLC_PREFIX=/opt/nec/ve/nlc/2.1.0
export MPI_PREFIX=/opt/nec/ve/mpi/2.9.0
. ${MPI_PREFIX}/bin/necmpivars.sh
. ${NLC_PREFIX}/bin/nlcvars.sh
```

### Build

```
mkdir build
cd build
../configure aurora
make
```

### Run

```
cd bin
./run.sh
```

### Rebuilding libvhcallVH.so

When rebuilding `libvhcallVH.so` you will need to use a modern gcc compiler, for example
from devtoolset-8 (which can be installed in CentOS 7).

Activate it with
```
source scl_source enable devtoolset-8
```

Then change into the libvhcall directory and type `make`.

