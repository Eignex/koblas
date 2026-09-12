#ifndef KOBLAS_KERNELS_H
#define KOBLAS_KERNELS_H
#include "koblas_probe.h"

/* Typed execution returns a status. Invalid/unavailable IDs are rejected before reading operands or
 * mutating output. The caller validates backing-buffer windows and operation shape; pointers are borrowed
 * for this call only and may alias as specified by the Kotlin kernel contract. No performance fallback. */
#define KOBLAS_OP_DENSE_DOT 1u
KOBLAS_API int32_t koblas_dense_dot_v1(uint32_t kernel_id, const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_SSQD 2u
KOBLAS_API int32_t koblas_dense_ssqd_v1(uint32_t kernel_id, const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_AXPY 3u
KOBLAS_API int32_t koblas_dense_axpy_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len);
#define KOBLAS_OP_DENSE_AXPY_ARITHMETIC 4u
KOBLAS_API int32_t koblas_dense_axpy_arithmetic_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len);
#define KOBLAS_OP_DENSE_SCALE 5u
KOBLAS_API int32_t koblas_dense_scale_v1(uint32_t kernel_id, double *v, int32_t v_off, double alpha, int32_t len);
#define KOBLAS_OP_DENSE_NRM2 6u
KOBLAS_API int32_t koblas_dense_nrm2_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_SUM 7u
KOBLAS_API int32_t koblas_dense_sum_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_ASUM 8u
KOBLAS_API int32_t koblas_dense_asum_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_IAMAX 9u
KOBLAS_API int32_t koblas_dense_iamax_v1(uint32_t kernel_id, const double *v, int32_t v_off, int32_t len, int32_t *result);
#define KOBLAS_OP_DENSE_GEMM_TILE 10u
KOBLAS_API int32_t koblas_dense_gemm_tile_v1(uint32_t kernel_id, int32_t depth, const double *packed_a, int32_t a_off, const double *packed_b, int32_t b_off, double *c, int32_t c_off, int32_t ldc);
#define KOBLAS_OP_DENSE_SWAP 11u
KOBLAS_API int32_t koblas_dense_swap_v1(uint32_t kernel_id, double *a, int32_t a_off, double *b, int32_t b_off, int32_t len);
#define KOBLAS_OP_DENSE_DOT4 12u
KOBLAS_API int32_t koblas_dense_dot4_v1(uint32_t kernel_id, const double *a, int32_t a_off, int32_t stride, const double *b, int32_t b_off, int32_t len, double *out, int32_t out_off);
#define KOBLAS_OP_DENSE_AXPY4 13u
KOBLAS_API int32_t koblas_dense_axpy4_v1(uint32_t kernel_id, double *y, int32_t y_off, const double *a, int32_t a_off, int32_t stride, double c0, double c1, double c2, double c3, int32_t len);
#define KOBLAS_OP_DENSE_DOT_AXPY 14u
KOBLAS_API int32_t koblas_dense_dot_axpy_v1(uint32_t kernel_id, double *y, int32_t y_off, double alpha, const double *a, int32_t a_off, const double *x, int32_t x_off, int32_t len, double *result);
#define KOBLAS_OP_DENSE_ROTM 15u
KOBLAS_API int32_t koblas_dense_rotm_v1(uint32_t kernel_id, double *x, int32_t x_off, int32_t x_stride, double *y, int32_t y_off, int32_t y_stride, int32_t len, double h11, double h12, double h21, double h22);
#define KOBLAS_OP_SPARSE_DOT_DENSE 16u
KOBLAS_API int32_t koblas_sparse_dot_dense_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, const double *dense, double *result);
#define KOBLAS_OP_SPARSE_DOT_SPARSE 17u
KOBLAS_API int32_t koblas_sparse_dot_sparse_v1(uint32_t kernel_id, const int32_t *a_indices, const double *a_values, int32_t a_len, const int32_t *b_indices, const double *b_values, int32_t b_len, double *result);
#define KOBLAS_OP_SPARSE_AXPY 18u
KOBLAS_API int32_t koblas_sparse_axpy_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, double alpha, double *dense);
#define KOBLAS_OP_SPARSE_SCATTER 19u
KOBLAS_API int32_t koblas_sparse_scatter_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, const double *values, int32_t value_off, int32_t len, double *dense);
#define KOBLAS_OP_SPARSE_NRM2 20u
KOBLAS_API int32_t koblas_sparse_nrm2_v1(uint32_t kernel_id, const int32_t *indices, int32_t index_off, int32_t len, const double *values, double *result);
#define KOBLAS_OP_SPARSE_GATHER 21u
KOBLAS_API int32_t koblas_sparse_gather_v1(uint32_t kernel_id, const int32_t *indices, double *values, int32_t len, const double *dense);
#define KOBLAS_OP_SPARSE_GATHER_ZERO 22u
KOBLAS_API int32_t koblas_sparse_gather_zero_v1(uint32_t kernel_id, const int32_t *indices, double *values, int32_t len, double *dense);
#define KOBLAS_OP_DENSE_TRSM_TILE 23u
KOBLAS_API int32_t koblas_dense_trsm_tile_v1(uint32_t kernel_id, int32_t valid_rows, int32_t order, const double *packed_triangle, int32_t triangle_off, int32_t lower, int32_t unit_diag, double *x, int32_t x_off);
#define KOBLAS_OP_DENSE_GEMM_TRSM_TILE 24u
KOBLAS_API int32_t koblas_dense_gemm_trsm_tile_v1(uint32_t kernel_id, int32_t depth, int32_t valid_rows, int32_t order, const double *packed_a, int32_t a_off, const double *packed_b, int32_t b_off, const double *packed_triangle, int32_t triangle_off, int32_t lower, int32_t unit_diag, double *x, int32_t x_off);
#endif
