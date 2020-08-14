#ifndef VEL_HPCG_KERNELS_HPP
#define VEL_HPCG_KERNELS_HPP


#include "Geometry.hpp"

extern "C" {
void intrin_gs_colwise(const local_int_t ics, const local_int_t ice, const double *a,
                       const double *idiag, const local_int_t lda, const local_int_t m,
                       const local_int_t *ja, double *xv, double *yv);

void intrin_gs_colwise_regs(const local_int_t ics, const local_int_t ice, const double *a,
                            const double *idiag, const local_int_t lda, const local_int_t m,
                            const local_int_t *ja, double *xv, double *yv);

void intrin_gs_colwise_casc(const local_int_t ics, const local_int_t ice, const double *a,
                            const double *idiag, const local_int_t lda, const local_int_t m,
                            const local_int_t *ja, double *xv, double *yv);

void intrin_gs_colwise_epilogue(const local_int_t ics, const local_int_t ice, const double *a,
                                const double *idiag, const local_int_t lda, const local_int_t m,
                                const local_int_t *ja, double *xv, double *yv);
}

#endif
