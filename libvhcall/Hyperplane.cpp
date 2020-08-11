#include <Geometry.hpp>
#include <iostream>

extern "C" void  Hyperplane(local_int_t nrow, int maxNonzerosInRow,
                            local_int_t *mtxIndL_, local_int_t *nonzerosInRow,
                            int *icolor)
{
  local_int_t  ** mtxIndL = new local_int_t*[nrow];
  local_int_t n = nrow;
  
  for (local_int_t i=0; i<nrow; ++i) {
    mtxIndL[i] = 0;
  }
  for (local_int_t i=0; i<nrow; ++i) {
    mtxIndL[i] = mtxIndL_ + i*maxNonzerosInRow;
  }

  ////////////////////////////////
  // Level Scheduling Algorithm //
  ////////////////////////////////

  for(local_int_t i=0; i<n; i++) icolor[i] = 0; // initialization

  for(local_int_t i=0; i<n; i++){
    local_int_t m=0;
    for(local_int_t jj=0; jj<nonzerosInRow[i]; jj++){
      local_int_t j = *(mtxIndL[i]+jj);
      if (j<i && m<icolor[j]) m=icolor[j];
    }
    icolor[i]=m+1;
  }
  // Hyperplanes lead to colors starting with 1.
  // Make color numbers start with 0.
  for(local_int_t i=0; i<n; i++)
    icolor[i]--;
  delete [] mtxIndL;
}
