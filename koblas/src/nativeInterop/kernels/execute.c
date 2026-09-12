#include "internal.h"

int32_t koblas_dense_dot_v1(uint32_t kernel_id, const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_DOT);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_dot_scalar(a, a_off, b, b_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_dot_sse2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_dot_avx2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_dot_neon(a, a_off, b, b_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_ssqd_v1(uint32_t kernel_id, const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_SSQD);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_ssqd_scalar(a, a_off, b, b_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_ssqd_sse2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_ssqd_avx2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_ssqd_neon(a, a_off, b, b_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_axpy_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_AXPY);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_axpy_scalar(y, y_off, alpha, x, x_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_axpy_sse2(y, y_off, alpha, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_axpy_avx2(y, y_off, alpha, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_axpy_neon(y, y_off, alpha, x, x_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_axpy_arithmetic_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_AXPY_ARITHMETIC);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_axpy_arithmetic_scalar(y, y_off, alpha, x, x_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_axpy_arithmetic_sse2(y, y_off, alpha, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_axpy_arithmetic_avx2(y, y_off, alpha, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_axpy_arithmetic_neon(y, y_off, alpha, x, x_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_scale_v1(uint32_t kernel_id, double *v, int32_t v_off, double alpha, int32_t len) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_SCALE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_scale_scalar(v, v_off, alpha, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_scale_sse2(v, v_off, alpha, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_scale_avx2(v, v_off, alpha, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_scale_neon(v, v_off, alpha, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_nrm2_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_NRM2);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_nrm2_scalar(v, v_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_nrm2_sse2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_nrm2_avx2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_nrm2_neon(v, v_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_sum_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_SUM);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_sum_scalar(v, v_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_sum_sse2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_sum_avx2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_sum_neon(v, v_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_asum_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_ASUM);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_asum_scalar(v, v_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_asum_sse2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_asum_avx2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_asum_neon(v, v_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_iamax_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, int32_t *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_IAMAX);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = -1; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_iamax_scalar(v, v_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_iamax_sse2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_iamax_avx2(v, v_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_iamax_neon(v, v_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_gemm_tile_v1(uint32_t kernel_id, int32_t depth, const double *packed_a, int32_t a_off, const double *packed_b, int32_t b_off, double *c, int32_t c_off, int32_t ldc) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_GEMM_TILE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (depth == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_gemm_tile_scalar(depth, packed_a, a_off, packed_b, b_off, c, c_off, ldc); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_gemm_tile_sse2(depth, packed_a, a_off, packed_b, b_off, c, c_off, ldc); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_gemm_tile_avx2(depth, packed_a, a_off, packed_b, b_off, c, c_off, ldc); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_gemm_tile_neon(depth, packed_a, a_off, packed_b, b_off, c, c_off, ldc); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_swap_v1(uint32_t kernel_id, double *a, int32_t a_off, double *b, int32_t b_off, int32_t len) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_SWAP);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_swap_scalar(a, a_off, b, b_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_swap_sse2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_swap_avx2(a, a_off, b, b_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_swap_neon(a, a_off, b, b_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_dot4_v1(uint32_t kernel_id, const double *a, int32_t a_off, int32_t stride, const double *b, int32_t b_off, int32_t len, double *out, int32_t out_off) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_DOT4);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) { for (int32_t i = 0; i < 4; i++) out[out_off + i] = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_dot4_scalar(a, a_off, stride, b, b_off, len, out, out_off); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_dot4_sse2(a, a_off, stride, b, b_off, len, out, out_off); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_dot4_avx2(a, a_off, stride, b, b_off, len, out, out_off); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_dot4_neon(a, a_off, stride, b, b_off, len, out, out_off); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_axpy4_v1(uint32_t kernel_id, double *y, int32_t y_off, const double *a, int32_t a_off, int32_t stride, double c0, double c1, double c2, double c3, int32_t len) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_AXPY4);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_axpy4_scalar(y, y_off, a, a_off, stride, c0, c1, c2, c3, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_axpy4_sse2(y, y_off, a, a_off, stride, c0, c1, c2, c3, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_axpy4_avx2(y, y_off, a, a_off, stride, c0, c1, c2, c3, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_axpy4_neon(y, y_off, a, a_off, stride, c0, c1, c2, c3, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_dot_axpy_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *a, int32_t a_off, const double *x, int32_t x_off, int32_t len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_DOT_AXPY);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_dense_dot_axpy_scalar(y, y_off, alpha, a, a_off, x, x_off, len); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: *result = koblas_dense_dot_axpy_sse2(y, y_off, alpha, a, a_off, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: *result = koblas_dense_dot_axpy_avx2(y, y_off, alpha, a, a_off, x, x_off, len); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: *result = koblas_dense_dot_axpy_neon(y, y_off, alpha, a, a_off, x, x_off, len); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_rotm_v1(uint32_t kernel_id, double *x, int32_t x_off, int32_t x_stride, double *y, int32_t y_off, int32_t y_stride, int32_t len, double h11, double h12, double h21, double h22) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_ROTM);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_rotm_scalar(x, x_off, x_stride, y, y_off, y_stride, len, h11, h12, h21, h22); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_rotm_sse2(x, x_off, x_stride, y, y_off, y_stride, len, h11, h12, h21, h22); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_rotm_avx2(x, x_off, x_stride, y, y_off, y_stride, len, h11, h12, h21, h22); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_rotm_neon(x, x_off, x_stride, y, y_off, y_stride, len, h11, h12, h21, h22); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_dot_dense_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, const double *dense, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_DOT_DENSE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_sparse_dot_dense_scalar(indices, index_off, values, value_off, len, dense); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_dot_sparse_v1(uint32_t kernel_id, const int32_t *a_indices, const double *a_values, int32_t a_len, const int32_t *b_indices, const double *b_values, int32_t b_len, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_DOT_SPARSE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_sparse_dot_sparse_scalar(a_indices, a_values, a_len, b_indices, b_values, b_len); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_axpy_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, double alpha, double *dense) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_AXPY);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_sparse_axpy_scalar(indices, index_off, values, value_off, len, alpha, dense); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_scatter_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, double *dense) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_SCATTER);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_sparse_scatter_scalar(indices, index_off, values, value_off, len, dense); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_nrm2_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, int32_t len, const double *values, double *result) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_NRM2);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (!result) return KOBLAS_INVALID_ARGUMENT;
    if (len == 0) { *result = 0.0; return KOBLAS_OK; }
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: *result = koblas_sparse_nrm2_scalar(indices, index_off, len, values); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_gather_v1(uint32_t kernel_id, const int32_t *indices, double *values, int32_t len, const double *dense) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_GATHER);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_sparse_gather_scalar(indices, values, len, dense); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_sparse_gather_zero_v1(uint32_t kernel_id, const int32_t *indices, double *values, int32_t len, double *dense) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_SPARSE_GATHER_ZERO);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (len == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_sparse_gather_zero_scalar(indices, values, len, dense); break;
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_trsm_tile_v1(uint32_t kernel_id, int32_t valid_rows, int32_t order, const double *packed_triangle, int32_t triangle_off, int32_t lower, int32_t unit_diag, double *x, int32_t x_off) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_TRSM_TILE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (valid_rows == 0 || order == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_trsm_tile_scalar(valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_trsm_tile_sse2(valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_trsm_tile_avx2(valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_trsm_tile_neon(valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}

int32_t koblas_dense_gemm_trsm_tile_v1(uint32_t kernel_id, int32_t depth, int32_t valid_rows, int32_t order, const double *packed_a, int32_t a_off, const double *packed_b, int32_t b_off, const double *packed_triangle, int32_t triangle_off, int32_t lower, int32_t unit_diag, double *x, int32_t x_off) {
    uint32_t reason = koblas_kernel_reason(kernel_id, KOBLAS_OP_DENSE_GEMM_TRSM_TILE);
    if (reason != KOBLAS_OK) return (int32_t)reason;
    if (valid_rows == 0 || order == 0) return KOBLAS_OK;
    switch (kernel_id % 16u) {
    case KOBLAS_SCALAR: koblas_dense_gemm_trsm_tile_scalar(depth, valid_rows, order, packed_a, a_off, packed_b, b_off, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#if defined(KOBLAS_BUILD_SSE2)
    case KOBLAS_SSE2: koblas_dense_gemm_trsm_tile_sse2(depth, valid_rows, order, packed_a, a_off, packed_b, b_off, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
#if defined(KOBLAS_BUILD_AVX2)
    case KOBLAS_AVX2: koblas_dense_gemm_trsm_tile_avx2(depth, valid_rows, order, packed_a, a_off, packed_b, b_off, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
#if defined(KOBLAS_BUILD_NEON)
    case KOBLAS_NEON: koblas_dense_gemm_trsm_tile_neon(depth, valid_rows, order, packed_a, a_off, packed_b, b_off, packed_triangle, triangle_off, lower, unit_diag, x, x_off); break;
#endif
    default: return KOBLAS_NOT_BUILT;
    }
    return KOBLAS_OK;
}
