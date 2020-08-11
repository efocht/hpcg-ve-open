
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation.hpp"
#include <cassert>
#include <iostream>

static int mg_level = 0;

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  //A.isMgOptimized = false;
  //return ComputeMG_ref(A, r, x);
  
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) {
      ierr += ComputeSYMGS(A, r, x, i ? 0 : 1);
    }
    if (ierr!=0) return(ierr);
    ierr = ComputeSPMV_L(A, x, *A.mgData->Axf);
    if (ierr!=0) return(ierr);
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);
    if (ierr!=0) return(ierr);
    mg_level++;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);
    mg_level--;
    if (ierr!=0) return(ierr);
    ierr = ComputeProlongation(A, x);
    if (ierr!=0) return(ierr);
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) {
      ierr += ComputeSYMGS(A, r, x, 0);
    }
    if (ierr!=0) return(ierr);
  }
  else {
    ierr += ComputeSYMGS(A, r, x, 0);

    if (ierr!=0) return(ierr);
  }
  return 0;
}
