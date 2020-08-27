//
// /opt/nec/nosupport/llvm-ve-1.15.0/bin/clang++ -fPIC --target=ve-linux -std=c++11 -mllvm -regalloc=basic -Rpass=unroll -O3 -S -o kernel.s -c kernel.cpp
//

#include "velintrin.h"
#include <cstdint>

#define UNR 6

typedef int32_t local_int_t;
const uint32_t max_vl = 256;

#define LOAD_Y(WORK,YP) do {                                    \
  if (blk < max_vl) gvl = blk;                                  \
  WORK = _vel_vldnc_vssvl(8, YP, WORK, gvl);                    \
  blk -= max_vl;                                                \
  if (blk > 0)                                                  \
    YP += max_vl;                                               \
  else                                                          \
    goto LOAD_Y_DONE;                                           \
  } while(0)

#define STORE_Y(WORK,YP) do {                                   \
  if (blk < max_vl) gvl = blk;                                  \
  _vel_vstncot_vssl(WORK, 8, (void *)YP, gvl);                  \
  blk -= max_vl;                                                \
  if (blk > 0)                                                  \
    YP += max_vl;                                               \
  else                                                          \
    goto STORE_Y_DONE;                                          \
  } while(0)

#define GATHER_ADD(WORK,XREG, AREG,XP,AP,JAP) do {              \
  if (blk < max_vl) gvl = blk;                                  \
  XREG = _vel_vldlsxnc_vssvl(4, JAP, XREG, gvl);                \
  AREG = _vel_vsfa_vvssvl(XREG, 3UL, (uint64_t)XP, AREG, gvl);          \
  XREG = _vel_vgt_vvssvl(AREG, (uint64_t)XP, 0, XREG, gvl);             \
  AREG = _vel_vldnc_vssvl(8, AP, AREG, gvl);                            \
  WORK = _vel_vfnmsbd_vvvvvl(WORK, AREG, XREG, WORK, gvl);              \
  blk -= max_vl;                                                        \
  if (blk > 0) {                                                        \
    AP += max_vl;                                                       \
    JAP += max_vl;                                                      \
  } else                                                                \
    goto GATHER_ADD_DONE;                                               \
  } while(0)



extern "C"
void intrin_gs_colwise_defs(const local_int_t ics, const local_int_t ice, const double *a,
                            const double *idiag, const local_int_t lda, const local_int_t m,
                            const local_int_t *ja, double *xv, double *yv)
{
  int64_t n = ice - ics;
  double *xp = (double *)xv, *yp, *ap, *idiagp;
  int32_t *jap;
  uint32_t gvl;

  __vr work_reg[UNR], a_reg[UNR], x_reg[UNR];

  for (uint64_t i = 0; i < n; i += UNR*max_vl) {
    int64_t blk_len = (i + UNR*max_vl <= n) ? (UNR*max_vl) : (n - i);
    int64_t blk = blk_len;

    yp = &yv[i];
    gvl = max_vl;
    LOAD_Y(work_reg[0], yp);
    LOAD_Y(work_reg[1], yp);
    LOAD_Y(work_reg[2], yp);
    LOAD_Y(work_reg[3], yp);
    LOAD_Y(work_reg[4], yp);
    LOAD_Y(work_reg[5], yp);

  LOAD_Y_DONE:
    
    for (uint64_t j = 0; j < m; j++) {
      ap = (double *)&a[i + lda * j]; jap = (int32_t *)&ja[i + lda * j];
      gvl = max_vl;
      blk = blk_len;

      GATHER_ADD(work_reg[0], x_reg[0], a_reg[0], xp, ap, jap);
      GATHER_ADD(work_reg[1], x_reg[1], a_reg[1], xp, ap, jap);
      GATHER_ADD(work_reg[2], x_reg[2], a_reg[2], xp, ap, jap);
      GATHER_ADD(work_reg[3], x_reg[3], a_reg[3], xp, ap, jap);
      GATHER_ADD(work_reg[4], x_reg[4], a_reg[4], xp, ap, jap);
      GATHER_ADD(work_reg[5], x_reg[5], a_reg[5], xp, ap, jap);

    GATHER_ADD_DONE:
      do {} while(0);

    } // j loop

    yp = &yv[i];
    gvl = max_vl;
    blk = blk_len;

    STORE_Y(work_reg[0], yp);
    STORE_Y(work_reg[1], yp);
    STORE_Y(work_reg[2], yp);
    STORE_Y(work_reg[3], yp);
    STORE_Y(work_reg[4], yp);
    STORE_Y(work_reg[5], yp);

  STORE_Y_DONE:
    do {} while(0);

  } // i lopp

  _vel_svob();
}
