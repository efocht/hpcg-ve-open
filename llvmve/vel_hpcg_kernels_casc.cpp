#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>

#define UNR16 16
#define UNR8 8
#define UNR4 4
#define UNR2 2
#define UNR1 1
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAXVL 256

const local_int_t max_vl = 256;
const local_int_t blksz16 = 16 * 256;
const local_int_t blksz8  =  8 * 256;
const local_int_t blksz4  =  4 * 256;
const local_int_t blksz2  =  2 * 256;
const local_int_t blksz1  =  1 * 256;

#define UNROLL_GS_BLOCK(UNROLL,VLEN) do {                               \
  yp = &yv[imin];                                                       \
  _Pragma("clang loop unroll(full)")                                    \
  for (local_int_t k = 0; k < UNROLL; k++) {                            \
    work_reg[k] = _vel_vldnc_vssl(8, yp, VLEN);                         \
    yp += VLEN;                                                         \
  }                                                                     \
  for (local_int_t j = 0; j < m; j++) {                                 \
    ap = (double *)&a[imin + lda * j];                                  \
    jap = (int32_t *)&ja[imin + lda * j];                               \
    _Pragma("clang loop unroll(full)")                                  \
    for (local_int_t k = 0; k < UNROLL; k++) {                          \
      __vr x_reg = _vel_vldlsxnc_vssl(4, jap, VLEN);                    \
      __vr a_reg = _vel_vsfa_vvssl(x_reg, 3UL, (uint64_t)xp, VLEN);     \
      x_reg = _vel_vgt_vvssl(a_reg, (uint64_t)xp, 0, VLEN);             \
      a_reg = _vel_vldnc_vssl(8, ap, VLEN);                             \
      work_reg[k] = _vel_vfnmsbd_vvvvl(work_reg[k], a_reg, x_reg, VLEN); \
      ap += VLEN; jap += VLEN;                                          \
    }                                                                   \
  }                                                                     \
  yp = &yv[imin];                                                       \
  _Pragma("clang loop unroll(full)")                                    \
  for (local_int_t k = 0; k < UNROLL; k++) {                            \
    _vel_vstncot_vssl(work_reg[k], 8, (void *)yp, VLEN);                \
    yp += VLEN;                                                         \
  }                                                                     \
  } while(0)


extern "C"
void intrin_gs_colwise_casc(const local_int_t ics, const local_int_t ice, const double *a,
                            const double *idiag, const local_int_t lda, const local_int_t m,
                            const local_int_t *ja, double *xv, double *yv)
{
  local_int_t n = ice - ics;
  double *xp = (double *)xv, *yp, *ap, *idiagp;
  int32_t *jap;
  uint32_t gvl;
  local_int_t imin, imax;
  __vr work_reg[UNR16];

  for (local_int_t i = 0; i < n; i += blksz16) {
    imin = i;
    imax = MIN(i + blksz16, n);
    if (imax < n || imax-imin >= blksz16) {
      UNROLL_GS_BLOCK(UNR16,MAXVL);
      continue;
    }
    if (imax-imin > blksz8) {
      UNROLL_GS_BLOCK(UNR8,MAXVL);
      imin += blksz8;
    }
    if (imax-imin > blksz4) {
      UNROLL_GS_BLOCK(UNR4,MAXVL);
      imin += blksz4;
    }
    if (imax-imin > blksz2) {
      UNROLL_GS_BLOCK(UNR2,MAXVL);
      imin += blksz2;
    }
    if (imax-imin > blksz1) {
      UNROLL_GS_BLOCK(UNR1,MAXVL);
      imin += blksz1;
    }
    gvl = imax - imin;
    UNROLL_GS_BLOCK(UNR1,gvl);
  }
  _vel_svob();

}
