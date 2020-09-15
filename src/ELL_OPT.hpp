#ifndef ELL_OPT_HPP
#define ELL_OPT_HPP

#include "Geometry.hpp"

//
// +--------+--------+  
// |        |        |
// |        |        |
// |   L    |   U    |  n
// |        |        | rows
// |        |        |
// |        |        |
// +--------+--------+
// |<- mL ->|<- mU ->|
//
// matrix stored column wise, therefore both, L and U are contiguous in memory

struct ELL_STRUCT{
  double       *a;
  local_int_t  *ja;
  local_int_t   lda;
  local_int_t   n;
  local_int_t   m;
  local_int_t   mL;     // width of lower matrix in MATRIX_LU format
  local_int_t   mU;     // width of upper matrix in MATRIX_LU format
};
typedef struct ELL_STRUCT ELL;

//
// +--------+  
// |        |
// |        |
// |        | nah
// |        | rows
// |        |
// |        |
// +--------+
// |<- mh ->|
//
// matrix stored column wise, therefore both, L and U are contiguous in memory

struct HALO_STRUCT{
  double          *ah;
  local_int_t    *jah;
  local_int_t   *rows;  // rows corresponding to each halo matrix row
  local_int_t  *hcptr;  // color pointer in halo matrix, for all colors
  local_int_t    ldah;  // leading dimension of halo matrix, usually nah
  local_int_t      nh;  // length of incoming/outgoing X values
  local_int_t     nah;  // length of AH matrix!
  local_int_t      mh;  // width of halo matrix
  double           *v;  // a vector for SPMV that should save alloc/free time
};
typedef struct HALO_STRUCT HALO;

struct OPT_STRUCT{
  ELL *ell;
  HALO *halo;
  double *diag;
  double *idiag;
  local_int_t *perm0;   // 0 origin
  local_int_t *iperm0;  // 0 origin
  local_int_t *icptr;   // pointer to start row of each color
  local_int_t *icolor;   // color of each row. freed when not needed any more
  local_int_t maxcolor;  // maximum color
  local_int_t *color_mL; // width of lower matrix in LU format for each color
  local_int_t *color_mU; // width of upper matrix in LU format for each color
  double * work1;
  double *work2;
};
typedef struct OPT_STRUCT OPT;

#endif
