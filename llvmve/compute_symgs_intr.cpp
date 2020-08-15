#include "compute_symgs_intr.hpp"
#if 0
/*
 * Triangular solve for ELL format with 0-base column indexing, colored sparse matrix.
 */ 
extern "C" void
ell_col_b0_trsv_intr(const local_int_t ic_min, const local_int_t ic_max,
                     const local_int_t *icptr, const local_int_t *iclen,
                     const double *a, const double *idiag,
                     const local_int_t lda, const local_int_t *ja,
                     double *xv, double *work)
{
  local_int_t ic, irs, ire, m;

  if (ic_min < ic_max) {
    for (local_int_t ic = ic_min; ic < ic_max; ic++) {
      irs = icptr[ic];		// start row for color ic
      ire = icptr[ic+1];	// end row for color ic
      m   = iclen[ic];		// max width of matrix for the ic color
      //
      // llvm fails to inline the function
      //
      ell_gs_colwise_sweep(irs, ire, a, idiag, lda, m, ja, xv, work);
    }
  } else {
    for (local_int_t ic = ic_max - 1; ic >= ic_min; ic--) {
      irs = icptr[ic];		// start row for color ic
      ire = icptr[ic+1];	// end row for color ic
      m   = iclen[ic];		// max width of matrix for the ic color
      //
      // llvm fails to inline the function
      //
      ell_gs_colwise_sweep(irs, ire, a, idiag, lda, m, ja, xv, work);
    }
  }
}
#endif

extern "C" void
ell_col_b0_trsv_up(const local_int_t ic_min, const local_int_t ic_max,
                   const local_int_t *icptr, const local_int_t *iclen,
                   const double *a, const double *idiag,
                   const local_int_t lda, const local_int_t *ja,
                   double *xv, double *yv)
{
  local_int_t ic, ics, ice, m;

  for (local_int_t ic = ic_min; ic < ic_max; ic++) {
    ics = icptr[ic];		// start row for color ic
    ice = icptr[ic+1];	// end row for color ic
    m   = iclen[ic];		// max width of matrix for the ic color

    local_int_t n = ice - ics;
    double *xp = (double *)xv, *yp, *ap, *idiagp;
    int32_t *jap;
    uint32_t vl;

#define UNR 20

    __vr work_reg[UNR], a_reg[UNR], x_reg[UNR];

    for (local_int_t i = 0; i < n; i += UNR*max_vl) {
      int blk_len = (i + UNR * max_vl <= n) ? (UNR * max_vl) : (n - i);
      int blk = blk_len;

      yp = &yv[ics];
      vl = max_vl;
#pragma clang loop unroll(full)
      for (local_int_t k = 0; k < UNR; k++) {
        if (blk < max_vl) vl = blk;
        work_reg[k] = _vel_vldnc_vssvl(8, yp, work_reg[k], vl);
        blk -= max_vl;
        yp += vl;
        if (blk <= 0) break;
      }

      for (local_int_t j = 0; j < m; j++) {
        ap = (double *)&a[ics + lda * j]; jap = (int32_t *)&ja[ics + lda * j];
        vl = max_vl;
        blk = blk_len;
#pragma clang loop unroll(full)
        for (local_int_t k = 0; k < UNR; k++) {
          if (blk < max_vl) vl = blk;
          x_reg[k] = _vel_vldlsxnc_vssvl(4, jap, x_reg[k], vl);
          a_reg[k] = _vel_vsfa_vvssvl(x_reg[k], 3UL, (uint64_t)xp, a_reg[k], vl);
          x_reg[k] = _vel_vgt_vvssvl(a_reg[k], (uint64_t)xp, 0, x_reg[k], vl);
          a_reg[k] = _vel_vldnc_vssvl(8, ap, a_reg[k], vl);
          work_reg[k] = _vel_vfnmsbd_vvvvvl(work_reg[k], a_reg[k], x_reg[k], work_reg[k], vl);
          blk -= max_vl;
          ap += vl; jap += vl;
          if (blk <= 0) break;
        } // k loop
      } // j loop

      yp = &yv[ics];
      vl = max_vl;
      blk = blk_len;
#pragma clang loop unroll(full)
      for (local_int_t k = 0; k < UNR; k++) {
        if (blk < max_vl) vl = blk;
        _vel_vstncot_vssl(work_reg[k], 8, (void *)yp, vl);
        blk -= max_vl;
        yp += vl;
        if (blk <= 0) break;
      } // k loop
    } // i lopp
#undef UNR
    _vel_svob();

#if 0
    vecmult_elemwise(ice - ics, (const double *)&yv[ics], &idiag[ics], &xv[ics]);
#else
    idiagp = (double *)&idiag[ics];
    xp = &xv[ics];
    yp = &yv[ics];
    for (local_int_t i = ics; i < ice; i += max_vl) {
      local_int_t imax = MIN(i + max_vl, ice);
      vl = imax - i;
      __vr yv = _vel_vldnc_vssl(8, yp, vl);
      __vr dv = _vel_vldnc_vssl(8, idiagp, vl);
      __vr xv = _vel_vfmuld_vvvl(yv, dv, vl);
      _vel_vstnc_vssl(xv, 8, xp, vl);
      yp += vl; idiagp += vl; xp += vl;
    }
    _vel_svob();
#endif
  }
}
