#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>
#include "stdio.h"

const local_int_t max_vl = 256;

void intrin_gs_colwise(const local_int_t ics, const local_int_t ice, const double *a,
    const double *idiag, const local_int_t lda, const local_int_t m,
    const local_int_t *ja, double *xv, double *work)
{
    for (local_int_t j = 0; j < m; j++)
    {
        for (local_int_t i = ics; i < ice; i+= max_vl)
        {
            uint32_t gvl = ((i + max_vl) > ice) ? (ice - i) : max_vl;
            // load a, ja, work
            __vr a_values = _vel_vld_vssl(8, &a[i + lda * j], gvl);
            __vr col_indices = _vel_vldu_vssl(4, &ja[i + lda * j], gvl); // 32-bit load
            __vr work_values = _vel_vld_vssl(8, &work[i], gvl);
            // x indirections and load
            __vr x_gather_addr = _vel_vsfa_vvssl(col_indices, 3UL, (uint64_t)xv, gvl); 
            // __vr x_values = _vel_vgt_vvssl(x_gather_addr, (uint64_t)&xv[0], (uint64_t)&xv[m+1], gvl); 
            __vr x_values = _vel_vld_vssl(8, &a[i + lda * j], gvl);
            // 
            work_values = _vel_vfmsbd_vvvvl(work_values, a_values, x_values, gvl); // work_values -= a[i + lda * j] * xv[ja[i + lda * j]];

            _vel_vst_vssl(work_values, 8, (void *)&work[i], gvl); // work[i] -= work_values;
        }
    }

    // for (local_int_t i = ics; i < ice; i+=256)
    // {
    //     __vr work_values = _vel_vld_vssl(8, &work[i], 256);
    //     __vr diag_values = _vel_vld_vssl(8, &idiag[i], 256); // 
    //     __vr x_values = _vel_vfmuld_vvvl(work_values, diag_values, 256);   // work[:] * idiag[:];
    //     _vel_vst_vssl(x_values, 8, &xv[i], 256); // xv[i] = work[i] * idiag[i];
    // }
    // _vel_svob();
}