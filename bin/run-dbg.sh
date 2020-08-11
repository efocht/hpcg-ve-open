#!/bin/bash

mkinput () {
    cat > hpcg.dat <<EOF
HPCG benchmark input file
NEC HPC Europe, SX-Aurora TSUBASA Vector Engine
$1 $2 $3
$4
EOF
}

HPCG=./xhpcg
HPCG="/usr/bin/xterm -e /opt/nec/ve/bin/gdb --args ./xhpcg"

#mkinput 56 216 376 60
mkinput 24 24 24 30

env LIBVHCALLVH=`pwd`/libvhcallVH.so /opt/nec/ve/bin/mpirun -ve 1 -np 2 /usr/bin/hugectl --heap $HPCG

