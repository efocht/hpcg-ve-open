
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

#ifndef EXCHANGEHALO_HPP
#define EXCHANGEHALO_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#ifndef HPCG_NO_MPI
#include <mpi.h>

void ExchangeHalo_nowait(const SparseMatrix & A, Vector & x, int is_first, MPI_Request **requests, int &num_requests);
void ExchangeHalo_wait(int is_first, MPI_Request **requests, int &num_requests);
void ExchangeHalo_probe(int is_first);
#endif // NO_MPI

void ExchangeHalo(const SparseMatrix & A, Vector & x, int is_first);


#endif // EXCHANGEHALO_HPP
