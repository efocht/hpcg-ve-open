NCC=ncc
NCFLAGS=

GCC=gcc
GCXX=g++
ICC=icc
ICXX=icpc
#GCC=/opt/nec/nosupport/llvm-ve-1.7.0/bin/clang -Rpass=loop-vectorize
#GCXX=/opt/nec/nosupport/llvm-ve-1.7.0/bin/clang++ -Rpass=loop-vectorize
#GCFLAGS=-g -I../src
GCFLAGS=-O3 -I../src -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512ifma -mavx512vbmi
GCFLAGS=-O3 -I../src -mavx2

ICFLAGS=-O3 -I../src -xCORE-AVX2 -qopt-zmm-usage=high -qopenmp -std=c++11

CC = $(GCC)
CXX = $(GCXX)
CFLAGS = $(GCFLAGS)

all: libvhcallVH.so

#libvhcallVE.o: libvhcallVE.c
#	$(NCC) -fPIC -c $(NCFLAGS) $^

libvhcallVH.so: libvhcallVH.o libvhcallVHcpp.o Hyperplane.o GetPerm.o
	$(CXX) -shared -fPIC -o $@ $(CFLAGS) -Wl,-zdefs -Wl,-zrelro -Wl,-znow $^

#       $(ICXX) -shared -fPIC -o $@ $(ICFLAGS) -Wl,-zdefs ${MKLROOT}/../compiler/lib/intel64/lib{imf,svml,irng,iomp5}.a $^

libvhcallVH.o: libvhcallVH.c
	$(CC) -fPIC -c $(CFLAGS) $^

libvhcallVHcpp.o: libvhcallVHcpp.cpp
	$(CXX) -fPIC -c $(CFLAGS) $^

Hyperplane.o: Hyperplane.cpp
	$(CXX) -fPIC -c $(CFLAGS) $^

GetPerm.o: GetPerm.cpp
	$(CXX) -fPIC -c $(CFLAGS) $^

clean:
	rm -f *.so *.o
