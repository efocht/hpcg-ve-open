//
// /opt/nec/nosupport/llvm-ve-1.15.0/bin/clang++ -fPIC --target=ve-linux -std=c++11 -mllvm -regalloc=basic -Rpass=unroll -O3 -S -o kernel.s -c kernel.cpp
//

#include "velintrin.h"
#include <cstdint>

#define UNR 20

typedef int32_t local_int_t;
const local_int_t max_vl = 256;

void intrin_gs_colwise_regs(const local_int_t ics, const local_int_t ice, const double *a,
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
      work_reg[k] = _vel_vldnc_vssvl(8, yp, work_reg[k], gvl);
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
        x_reg[k] = _vel_vldlsxnc_vssvl(4, jap, x_reg[k], gvl);
        a_reg[k] = _vel_vsfa_vvssvl(x_reg[k], 3UL, (uint64_t)xp, a_reg[k], gvl);
        x_reg[k] = _vel_vgt_vvssvl(a_reg[k], (uint64_t)xp, 0, x_reg[k], gvl);
        a_reg[k] = _vel_vldnc_vssvl(8, ap, a_reg[k], gvl);
        work_reg[k] = _vel_vfnmsbd_vvvvvl(work_reg[k], a_reg[k], x_reg[k], work_reg[k], gvl);
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
  } // i lopp

  _vel_svob();
}
