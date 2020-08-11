
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

#ifndef MYTIMER_HPP
#define MYTIMER_HPP

#ifndef HPCG_NO_MPI
#include <mpi.h>

static inline double mytimer(void) {
  return MPI_Wtime();
}

#elif !defined(HPCG_NO_OPENMP)

// If this routine is compiled with HPCG_NO_MPI defined and not compiled with HPCG_NO_OPENMP then use the OpenMP timer
#include <omp.h>
static inline double mytimer(void) {
  return omp_get_wtime();
}

#else

#include <cstdint>
// the return value has to be divided by 1e9 to get seconds
static inline double mytimer() {
  uint64_t ret;
  void *vehva = ((void *)0x000000001000);
  asm volatile("lhm.l %0,0(%1)":"=r"(ret):"r"(vehva));
  //the "800" is due to the base frequency of Tsubasa
  return (double)(((uint64_t)1000 * ret) / 800) / 1e9;
}
#endif

#endif // MYTIMER_HPP
