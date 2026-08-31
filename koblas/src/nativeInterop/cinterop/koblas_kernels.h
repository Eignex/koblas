#ifndef KOBLAS_KERNELS_H
#define KOBLAS_KERNELS_H

#include <math.h>
#include <stdint.h>

#if defined(KOBLAS_KERNELS_IMPLEMENTATION)
#define KOBLAS_KERNEL __attribute__((visibility("default")))
#else
#define KOBLAS_KERNEL static inline
#endif

KOBLAS_KERNEL double koblas_dense_dot(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
    double sum = 0.0;
    for (int32_t i = 0; i < len; i++) sum += a[a_off + i] * b[b_off + i];
    return sum;
}

KOBLAS_KERNEL void koblas_dense_axpy(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    if (alpha == 0.0) return;
    for (int32_t i = 0; i < len; i++) y[y_off + i] += alpha * x[x_off + i];
}

KOBLAS_KERNEL void koblas_dense_axpy_arithmetic(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    for (int32_t i = 0; i < len; i++) y[y_off + i] += alpha * x[x_off + i];
}

KOBLAS_KERNEL void koblas_dense_scale(double *v, int32_t v_off, double alpha, int32_t len) {
    if (alpha == 1.0) return;
    for (int32_t i = 0; i < len; i++) v[v_off + i] *= alpha;
}

KOBLAS_KERNEL double koblas_dense_nrm2(const double *v, int32_t v_off, int32_t len) {
    double squares = 0.0;
    for (int32_t i = 0; i < len; i++) {
        const double value = v[v_off + i];
        squares += value * value;
    }
    if (isfinite(squares) && squares >= 0x1p-1022) return sqrt(squares);

    double maximum = 0.0;
    for (int32_t i = 0; i < len; i++) {
        const double magnitude = fabs(v[v_off + i]);
        if (magnitude > maximum) maximum = magnitude;
    }
    if (maximum == 0.0 || isinf(maximum)) return sqrt(squares);

    double scaled_squares = 0.0;
    for (int32_t i = 0; i < len; i++) {
        const double scaled = v[v_off + i] / maximum;
        scaled_squares += scaled * scaled;
    }
    return maximum * sqrt(scaled_squares);
}

KOBLAS_KERNEL double koblas_dense_asum(const double *v, int32_t v_off, int32_t len) {
    double sum = 0.0;
    for (int32_t i = 0; i < len; i++) sum += fabs(v[v_off + i]);
    return sum;
}

KOBLAS_KERNEL void koblas_dense_swap(
    double *a, int32_t a_off, double *b, int32_t b_off, int32_t len
) {
    for (int32_t i = 0; i < len; i++) {
        const double temporary = a[a_off + i];
        a[a_off + i] = b[b_off + i];
        b[b_off + i] = temporary;
    }
}

KOBLAS_KERNEL void koblas_dense_dot4(
    const double *a, int32_t a_off, int32_t stride, const double *b, int32_t b_off,
    int32_t len, double *out, int32_t out_off
) {
    double r0 = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    double r3 = 0.0;
    for (int32_t i = 0; i < len; i++) {
        const double bi = b[b_off + i];
        r0 += a[a_off + i] * bi;
        r1 += a[a_off + stride + i] * bi;
        r2 += a[a_off + 2 * stride + i] * bi;
        r3 += a[a_off + 3 * stride + i] * bi;
    }
    out[out_off] = r0;
    out[out_off + 1] = r1;
    out[out_off + 2] = r2;
    out[out_off + 3] = r3;
}

KOBLAS_KERNEL double koblas_sparse_dot_dense(
    const int32_t *indices, const double *values, int32_t len, const double *dense
) {
    double sum = 0.0;
    for (int32_t k = 0; k < len; k++) sum += values[k] * dense[indices[k]];
    return sum;
}

KOBLAS_KERNEL double koblas_sparse_dot_sparse(
    const int32_t *a_indices, const double *a_values, int32_t a_len,
    const int32_t *b_indices, const double *b_values, int32_t b_len
) {
    double sum = 0.0;
    int32_t a = 0;
    int32_t b = 0;
    while (a < a_len && b < b_len) {
        const int32_t ai = a_indices[a];
        const int32_t bi = b_indices[b];
        if (ai < bi) {
            a++;
        } else if (ai > bi) {
            b++;
        } else {
            sum += a_values[a] * b_values[b];
            a++;
            b++;
        }
    }
    return sum;
}

KOBLAS_KERNEL void koblas_sparse_axpy(
    const int32_t *indices, const double *values, int32_t len, double alpha, double *dense
) {
    if (alpha == 0.0) return;
    for (int32_t k = 0; k < len; k++) dense[indices[k]] += alpha * values[k];
}

KOBLAS_KERNEL void koblas_sparse_scatter(
    const int32_t *indices, const double *values, int32_t len, double *dense
) {
    for (int32_t k = 0; k < len; k++) dense[indices[k]] = values[k];
}

KOBLAS_KERNEL void koblas_sparse_gather(
    const int32_t *indices, double *values, int32_t len, const double *dense
) {
    for (int32_t k = 0; k < len; k++) values[k] = dense[indices[k]];
}

KOBLAS_KERNEL void koblas_sparse_gather_zero(
    const int32_t *indices, double *values, int32_t len, double *dense
) {
    for (int32_t k = 0; k < len; k++) {
        const int32_t index = indices[k];
        values[k] = dense[index];
        dense[index] = 0.0;
    }
}

#undef KOBLAS_KERNEL
#endif
