#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>

#define UNR 4

const local_int_t max_vl = 256;

extern "C"
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
