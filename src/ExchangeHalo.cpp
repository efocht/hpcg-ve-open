
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

/*!
 @file ExchangeHalo.cpp

 HPCG routine
 */

// Compile this routine only if running with MPI
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "Geometry.hpp"
#include "ExchangeHalo.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

#define MPI_MY_TAG 99

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
  @param[in]  is_first Flag signalling that this is the first call of SYMGS in the V cycle, i.e. the Halos are zero

 */
void ExchangeHalo(const SparseMatrix & A, Vector & x, int is_first)
{

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  double * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!num_neighbors) return;
  
  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  MPI_Request * request = new MPI_Request[2*num_neighbors];

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    if (!is_first)
      MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+i);
    else
      std::memset(x_external, 0, sizeof(double)*n_recv);
    x_external += n_recv;
  }


  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  if (!is_first) {
    for (int i = 0; i < num_neighbors; i++) {
      local_int_t n_send = sendLength[i];
      //MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
      MPI_Isend(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request+num_neighbors+i);
      sendBuffer += n_send;
    }
  }

  //
  // Complete the reads issued above
  //
  if (!is_first) {
#if 0
    MPI_Status status;
    // TODO: Thread this loop
    for (int i = 0; i < num_neighbors; i++) {
      if (MPI_Wait(request+i, &status) ) {
        std::exit(-1); // TODO: have better error exit
      }
    }
#else
    MPI_Status status[2*num_neighbors];
    if (MPI_Waitall(2*num_neighbors, request, status) ) {
      std::exit(-1); // TODO: have better error exit
    }
#endif
  }

  delete [] request;

  return;
}

/*!
  Communicates data that is at the border of the part of the domain assigned to this processor.

  @param[in]    A The known system matrix
  @param[inout] x On entry: the local vector entries followed by entries to be communicated; on exit: the vector with non-local entries updated by other processors
  @param[in]  is_first Flag signalling that this is the first call of SYMGS in the V cycle, i.e. the Halos are zero

 */

void ExchangeHalo_nowait(const SparseMatrix & A, Vector & x, int is_first, MPI_Request **requests, int &num_requests)
{

  // Extract Matrix pieces

  local_int_t localNumberOfRows = A.localNumberOfRows;
  int num_neighbors = A.numberOfSendNeighbors;
  local_int_t * receiveLength = A.receiveLength;
  local_int_t * sendLength = A.sendLength;
  int * neighbors = A.neighbors;
  double * sendBuffer = A.sendBuffer;
  local_int_t totalToBeSent = A.totalToBeSent;
  local_int_t * elementsToSend = A.elementsToSend;

  double * const xv = x.values;

  int size, rank; // Number of MPI processes, My process ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  num_requests = 2 * num_neighbors;
  if (!num_neighbors) return;
  
  //
  //  first post receives, these are immediate receives
  //  Do not wait for result to come, will do that at the
  //  wait call below.
  //

  *requests = new MPI_Request[2*num_neighbors];

  //
  // Externals are at end of locals
  //
  double * x_external = (double *) xv + localNumberOfRows;

  // Post receives first
  // TODO: Thread this loop
  for (int i = 0; i < num_neighbors; i++) {
    local_int_t n_recv = receiveLength[i];
    if (!is_first)
      MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, *requests+i);
    else
      std::memset(x_external, 0, sizeof(double)*n_recv);
    x_external += n_recv;
  }

  //
  // Fill up send buffer
  //

  // TODO: Thread this loop
  for (local_int_t i=0; i<totalToBeSent; i++) sendBuffer[i] = xv[elementsToSend[i]];

  //
  // Send to each neighbor
  //

  // TODO: Thread this loop
  if (!is_first) {
    for (int i = 0; i < num_neighbors; i++) {
      local_int_t n_send = sendLength[i];
      //MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
      MPI_Isend(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, *requests+num_neighbors+i);
      sendBuffer += n_send;
    }
  }
  return;
}
  
void ExchangeHalo_wait(int is_first, MPI_Request **requests, int &num_requests)
{  
  //
  // Complete the reads issued above
  //
  if (num_requests == 0)
    return;
  if (!is_first) {
    MPI_Status status[num_requests];
    if (MPI_Waitall(num_requests, *requests, status) ) {
      std::cout << "MPI_Waitall failed.\n";
      std::exit(-1); // TODO: have better error exit
    }
  }
  delete [] *requests;
  return;
}

void ExchangeHalo_probe(int is_first)
{  
  //
  // Just drive communication forward
  //
  if (!is_first) {
    MPI_Status status;
    int flag;
    //MPI_Request req = MPI_REQUEST_NULL;
    //if (MPI_Test(&req, &flag, &status) != MPI_SUCCESS) {
    if (MPI_Iprobe(MPI_ANY_SOURCE, MPI_MY_TAG, MPI_COMM_WORLD, &flag, 
                   &status) != MPI_SUCCESS) {
      std::cout << "MPI_Test failed.\n";
      std::exit(-1); // TODO: have better error exit
    }
  }
  return;
}

#endif
// ifndef HPCG_NO_MPI
