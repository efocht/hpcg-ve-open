int ComputeSYMGS_omp( const SparseMatrix & A, const Vector & r, Vector & x, int is_first) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  //return ComputeSYMGS_ref(A, r, x);
 
   assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  OPT* opt         = (OPT*)(A.optimizationData);
  const ELL*    ell      = opt->ell;
  const double* a        = ell->a;
  const local_int_t* ja  = ell->ja;
  const local_int_t lda  = ell->lda;
  const local_int_t m    = ell->m;
#ifdef WEIRDTEST
  const local_int_t mL   = ell->mL;
  const local_int_t mU   = ell->mU;
#endif
  const local_int_t n    = ell->n;
  const double alpha     = -1.0;
  const double beta      = +1.0;
  const double*  diag    = opt->diag;
  double*       idiag    = opt->idiag;
  double*       work1    = opt->work1;
  double*       work2    = opt->work2;
  const local_int_t* icptr    = opt->icptr;
  const local_int_t  maxcolor = opt->maxcolor;
  const local_int_t  minhalocolor = opt->minhalocolor;
  
  int exchg_calls = 1;

  const double * const rv = r.values;
  double * const xv = x.values;
  local_int_t ics, ice, nn;
  local_int_t color = 0;
  omp_barrier_type barr;
  int nt = omp_get_max_threads();

  
  /////////////////////////
  // Now the forward sweep.
  /////////////////////////

  omp_barrier_init(&barr, nt - 1);

#pragma omp parallel default(shared) private(ics,ice)
  {

    local_int_t ics, ice;         // redeclare here, we don't want any private()
    local_int_t ist, iend, ilen, inc;
    local_int_t ic = color;
    int  t = omp_get_thread_num();

    inc = (n + nt - 1) / nt;
    ist = t*inc;
#pragma omp for schedule(static)
    for(local_int_t i=0; i<n; i++){
      work1[i] = work2[i] = rv[i];
    }

#ifndef HPCG_NO_MPI
#ifdef HYPERCUBE

    if (t == 0) {

      ExchangeHalo(A, x, is_first);
      exchg_calls = 0;

    } else {
      // overlap halo exchange with computation, thread #0 does the exchange
      while (exchg_calls > 0 && ic < minhalocolor) {

        ics = icptr[ic];
        ice = icptr[ic+1];
        inc = (ice - ics + nt - 2) / (nt - 1);
        for (ist=ics+(t-1)*inc; ist<ice; ist+=(nt-1)*inc) {
          iend = MIN(ist + inc, ice);
          ilen = iend - ist;
          gs_block_sweep(ist, iend, a, idiag, lda, m, ja, xv, work1);
        }
        // this is a "manual" barrier for all threads but #0
        omp_barrier(&barr);
        ic++;
        if (t == 1)
          color = ic;
      }
    }

#else // HYPERCUBE
    if (t == 0) {
      ExchangeHalo(A, x, is_first);
      exchg_calls = 0;
    }
#endif // HYPERCUBE
#endif // HPCG_NO_MPI

#pragma omp barrier
    
    //ic = color;            // thread #0 must know where we are
    
    for (ic = color; ic < maxcolor; ic++) {
      ics = icptr[ic];
      ice = icptr[ic+1];
      inc = (ice - ics + nt - 1) / nt;

#pragma omp for schedule(static,1)
      for (ist=ics; ist<ice; ist+=inc) {
        iend = MIN(ist + inc, ice);
        ilen = iend - ist;

        gs_block_sweep(ist, iend, a, idiag, lda, m, ja, xv, work1);

      }
    }


  //////////////////////
  // Now the back sweep.
  //////////////////////

    for (ic = maxcolor - 1; ic >= 0; ic--) {
      ics = icptr[ic];
      ice = icptr[ic+1];
      inc = (ice - ics + nt - 1) / nt;
#pragma omp for schedule(static,1)
      for (ist=ics; ist<ice; ist+=inc) {
        iend = MIN(ist + inc, ice);
        ilen = iend - ist;
        gs_block_sweep(ist, iend, a, idiag, lda, m, ja, xv, work2);
      }
    }
  }   // end omp parallel block
  return 0;
}
