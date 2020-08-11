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

mkinput 56 216 376 120
#mkinput 24 24 24 30

export VE_PERF_MODE=VECTOR-MEM
veperf -d 1 > hpcg_veperf.out &
VEPPID=$!
sleep 1

env LIBVHCALLVH=`pwd`/libvhcallVH.so /opt/nec/ve/bin/mpirun -ve 1 -np 8 /usr/bin/hugectl --heap $HPCG

if [ -n "$VEPPID" ]; then
    sleep 2
    kill $VEPPID
fi
