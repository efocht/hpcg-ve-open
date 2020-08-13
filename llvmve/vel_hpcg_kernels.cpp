#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>

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

  __vr work_reg[UNR], a_reg[UNR], x_reg[UNR];
    
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
        x_reg[k] = _vel_vldlsxnc_vssvl(4, jap, x_reg[k], gvl); // 32-bit load
        // x indirections and gather
        __vr a_reg = _vel_vsfa_vvssvl(x_reg[k], 3UL, (uint64_t)xp, a_reg, gvl); // reuse a_values for gather addresses
        x_reg[k] = _vel_vgt_vvssvl(a_reg, (uint64_t)xp, 0, x_reg[k], gvl);
        // load a
        a_reg = _vel_vldnc_vssvl(8, ap, a_reg, gvl);
        // work -= a[i + lda * j] * xv[ja[i + lda * j]];
        work_reg[k] = _vel_vfnmsbd_vvvvvl(work_reg[k], a_reg, x_reg[k], work_reg[k], gvl);
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


void intrin_gs_colwise_epilogue(const local_int_t ics, const local_int_t ice, const double *a,
    const double *idiag, const local_int_t lda, const local_int_t m,
    const local_int_t *ja, double *xv, double *yv)
{
  local_int_t n = ice - ics;
  double *xp = (double *)xv, *yp, *ap, *idiagp;
  int32_t *jap;
  uint32_t gvl;
  __vr work_reg[UNR], a_reg[UNR], x_reg[UNR];

  for (local_int_t i = 0; i < ( n - (UNR*max_vl - 1)); i += (UNR*max_vl)) {

    yp = &yv[i];
#pragma clang loop unroll(full)
    for (local_int_t k = 0; k < UNR; k++) {
      work_reg[k] = _vel_vldnc_vssl(8, yp, max_vl);
      yp += max_vl;
    }

    for (local_int_t j = 0; j < m; j++) {
      ap = (double *)&a[i + lda * j]; 
      jap = (int32_t *)&ja[i + lda * j];
      
      #pragma clang loop unroll(full)
      for (local_int_t k = 0; k < UNR; k++) {
        // load ja
        x_reg[k] = _vel_vldlsxnc_vssvl(4, jap, x_reg[k], max_vl); // 32-bit load
        // x indirections and gather
        __vr a_reg = _vel_vsfa_vvssvl(x_reg[k], 3UL, (uint64_t)xp, a_reg, max_vl); // reuse a_values for gather addresses
        x_reg[k] = _vel_vgt_vvssvl(a_reg, (uint64_t)xp, 0, x_reg[k], max_vl);
        // load a
        a_reg = _vel_vldnc_vssvl(8, ap, a_reg, max_vl);
        // work -= a[i + lda * j] * xv[ja[i + lda * j]];
        work_reg[k] = _vel_vfnmsbd_vvvvvl(work_reg[k], a_reg, x_reg[k], work_reg[k], max_vl);
        //
        ap += max_vl; jap += max_vl;
      } // k loop
    } // j loop

    yp = &yv[i];
#pragma clang loop unroll(full)
    for (local_int_t k = 0; k < UNR; k++) {
      _vel_vstncot_vssl(work_reg[k], 8, (void *)yp, max_vl);
      yp += max_vl;
    } // k loop

  } // i loop

  // EPILOGUE
  local_int_t start_row = n - ( n % (UNR*max_vl));

  for (local_int_t i = start_row; i < n; i += max_vl) {
    
    int gvl = (i + max_vl < n) ? max_vl : (n - i);
    yp = &yv[i];
    work_reg[0] = _vel_vldnc_vssl(8, yp, gvl);

    for (local_int_t j = 0; j < m; j++) {
      ap = (double *)&a[i + lda * j]; 
      jap = (int32_t *)&ja[i + lda * j];
        // load ja
        x_reg[0] = _vel_vldlsxnc_vssvl(4, jap, x_reg[0], gvl); // 32-bit load
        // x indirections and gather
        __vr a_reg = _vel_vsfa_vvssvl(x_reg[0], 3UL, (uint64_t)xp, a_reg, gvl); // reuse a_values for gather addresses
        x_reg[0] = _vel_vgt_vvssvl(a_reg, (uint64_t)xp, 0, x_reg[0], gvl);
        // load a
        a_reg = _vel_vldnc_vssvl(8, ap, a_reg, gvl);
        // work -= a[i + lda * j] * xv[ja[i + lda * j]];
        work_reg[0] = _vel_vfnmsbd_vvvvvl(work_reg[0], a_reg, x_reg[0], work_reg[0], gvl);
        //
        ap += gvl; jap += gvl;
    } // j loop

    _vel_vstncot_vssl(work_reg[0], 8, (void *)yp, gvl);

  } // i loop

  _vel_svob();


}
