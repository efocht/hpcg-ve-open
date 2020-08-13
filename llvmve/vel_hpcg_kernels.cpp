#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>
#include "stdio.h"

#define UNR 3

const local_int_t max_vl = 256;

void intrin_gs_colwise(const local_int_t ics, const local_int_t ice, const double *a,
    const double *idiag, const local_int_t lda, const local_int_t m,
    const local_int_t *ja, double *xv, double *yv)
{
  local_int_t n = ice - ics;
  double *xp = (double *)xv, *yp, *ap, *idiagp;
  int32_t *jap;
  uint32_t gvl;

  __vr work_reg[UNR];
    
  for (local_int_t i = 0; i < n; i += UNR*max_vl) {
    int blk_len = (i + UNR * max_vl <= n) ? (UNR * max_vl) : (n - i);
    int blk = blk_len;

    yp = &yv[i];
    gvl = max_vl;
#pragma clang loop unroll(full)
    for (local_int_t k = 0; k < UNR; k++) {
      if (blk < max_vl) gvl = blk;
      work_reg[k] = _vel_vldnc_vssl(8, yp, gvl);
      blk -= max_vl;
      yp += gvl;
      if (blk <= 0) break;
    }

    for (local_int_t j = 0; j < m; j++) {
      ap = (double *)&a[i + lda * j]; jap = (int32_t *)&ja[i + lda * j];
      gvl = max_vl;
      blk = blk_len;
#pragma clang loop unroll(full)
      for (local_int_t k = 0; k < UNR; k++) {
        if (blk < max_vl) gvl = blk;
        // load ja
        __vr ix_reg = _vel_vldlsxnc_vssl(4, jap, gvl); // 32-bit load
        // x indirections and gather
        __vr addr_reg = _vel_vsfa_vvssl(ix_reg, 3UL, (uint64_t)xp, gvl); // reuse a_values for gather addresses
        __vr x_reg = _vel_vgt_vvssl(addr_reg, (uint64_t)xp, 0, gvl);
        // load a
        __vr a_reg = _vel_vldnc_vssl(8, ap, gvl);
        // work -= a[i + lda * j] * xv[ja[i + lda * j]];
        work_reg[k] = _vel_vfnmsbd_vvvvl(work_reg[k], a_reg, x_reg, gvl);
        //
        blk -= max_vl;
        ap += gvl; jap += gvl;
        if (blk <= 0) break;
      } // k loop
      
    } // j loop

  
    yp = &yv[i];
    gvl = max_vl;
    blk = blk_len;
#pragma clang loop unroll(full)
    for (local_int_t k = 0; k < UNR; k++) {
      if (blk < max_vl) gvl = blk;
      _vel_vstncot_vssl(work_reg[k], 8, (void *)yp, gvl);
      blk -= max_vl;
      yp += gvl;
      if (blk <= 0) break;
    } // k loop

    _vel_svob();
#if 0
    xp = &xv[ics + i];
    idiagp = (double *)&idiag[i];
    gvl = max_vl;
    blk = blk_len;
#pragma clang loop unroll(full)
    for (local_int_t k = 0; k < UNR; k++) {
      if (blk < max_vl) gvl = blk;
      __vr diag_reg = _vel_vldnc_vssl(8, idiagp, gvl);
      __vr x_val = _vel_vfmuld_vvvl(work_reg[k], diag_reg, gvl);   // work[:] * idiag[:];
      _vel_vstnc_vssl(x_val, 8, xp, gvl);                          // xv[i] = work[i] * idiag[i];
      blk -= max_vl;
      xp += gvl;
      idiagp += gvl;
      if (blk <= 0) break;
    } // k loop
#endif
  } // i loop

  // for (local_int_t i = ics; i < ice; i+=256)
  // {
  //     __vr work_values = _vel_vld_vssl(8, &work[i], 256);
  //     __vr diag_values = _vel_vld_vssl(8, &idiag[i], 256); // 
  //     __vr x_values = _vel_vfmuld_vvvl(work_values, diag_values, 256);   // work[:] * idiag[:];
  //     _vel_vst_vssl(x_values, 8, &xv[i], 256); // xv[i] = work[i] * idiag[i];
  // }

  _vel_svob();
}
