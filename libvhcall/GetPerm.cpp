#include <Geometry.hpp>
#include <iostream>

extern "C" void  GetPerm(local_int_t n, int maxcolor, int *icolor, local_int_t *perm0, local_int_t *icptr)
{
  for (local_int_t i=0; i<maxcolor+1; i++)
    icptr[i]=0;
  for (local_int_t i=0; i<n; i++) {
    // Count up for each color
    // Colors are now starting at zero
    ++icptr[icolor[i]+1];
  }

  for (local_int_t ic=1; ic<maxcolor+1; ic++) icptr[ic] += icptr[ic-1];

  //
  //  Make the permutation matrices perm[] and iperm[]
  //
  local_int_t* itmp = new local_int_t[maxcolor];
  for (local_int_t ic=0; ic<maxcolor; ic++)
    itmp[ic] = icptr[ic];
  for (local_int_t i=0; i<n; i++) {
    local_int_t ic = icolor[i];    // colors are starting at zero
    //perm[i]= itmp[ic]+1;           // 1 origin
    perm0[i]= itmp[ic];              // 0 origin
    itmp[ic]++;
  }
  delete [] itmp;
}


