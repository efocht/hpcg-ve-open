#include "velintrin.h"
#include "Geometry.hpp"
#include <cstdint>
#include "stdio.h"

const local_int_t max_vl = 256;

void intrin_gs_colwise(const local_int_t ics, const local_int_t ice, const double *a,
    const double *idiag, const local_int_t lda, const local_int_t m,
    const local_int_t *ja, double *xv, double *work)
{
    local_int_t n = ice - ics;

    for (local_int_t j = 0; j < m; j++)
    {
        for (local_int_t i = 0; i < n; i+= max_vl)
        {
            uint32_t gvl = ((i + max_vl) > n) ? (n - i) : max_vl;
            // load ja, work
            __vr x_values = _vel_vldlsxnc_vssl(4, &ja[i + lda * j], gvl); // 32-bit load  // reusing for column indices
            __vr work_values = _vel_vldnc_vssl(8, &work[i], gvl);
            // x indirections and gather
            __vr a_values = _vel_vsfa_vvssl(x_values, 3UL, (uint64_t)xv, gvl); // reuse a_values for gather addresses
            x_values = _vel_vgt_vvssl(a_values, (uint64_t)&xv[0], 0, gvl);
            // load a
            a_values = _vel_vldnc_vssl(8, &a[i + lda * j], gvl);
            // 
            work_values = _vel_vfnmsbd_vvvvl(work_values, a_values, x_values, gvl); // work_values -= a[i + lda * j] * xv[ja[i + lda * j]];

            _vel_vstnc_vssl(work_values, 8, (void *)&work[i], gvl); // work[i] -= work_values;
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
