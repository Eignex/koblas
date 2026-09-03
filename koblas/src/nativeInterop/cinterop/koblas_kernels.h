#ifndef KOBLAS_KERNELS_H
#define KOBLAS_KERNELS_H

#include <math.h>
#include <stdint.h>

#if defined(KOBLAS_KERNELS_IMPLEMENTATION)
#define KOBLAS_KERNEL __attribute__((visibility("default")))
#else
#define KOBLAS_KERNEL static inline
#endif

/*
 * Every reduction here carries four independent accumulators rather than one. A single accumulator chains
 * each add onto the previous result, and no compiler may reassociate a floating point sum unless asked, so
 * such a loop runs at add latency and vectorises to nothing at all. Four independent chains are
 * reassociated in the source, which is the author's call to make rather than the compiler's, so it needs
 * no flag and leaves the vectoriser free to pack them. koblas_dense_dot4 has carried four by construction
 * from the start and was the only reduction here that compiled to packed adds.
 *
 * Element-wise kernels need none of this: with no reduction to reassociate they already vectorise at -O3.
 * Two families cannot be helped this way and are left plain, each noted at its definition: rotm, whose
 * operands are strided and may overlap, and the sparse kernels, which index indirectly where the baseline
 * instruction set has neither gather nor scatter.
 *
 * Short runs take the plain tail instead, guarded by KOBLAS_UNROLL_MIN. Four chains have to be started and
 * then combined, and below a few dozen elements that fixed cost outweighs the shortened dependency chain.
 * Timing the two forms against each other here puts the crossover between 16 and 32 elements, and the JVM
 * SIMD kernels reached 32 by their own measurement, so the two implementations agree on the number. Since
 * the accumulators are still zero on that path, the combine folds to nothing and a short run returns
 * exactly what the plain loop alone returned.
 */
#define KOBLAS_UNROLL_MIN 32

KOBLAS_KERNEL double koblas_dense_dot(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; i + 4 <= len; i += 4) {
        s0 += a[a_off + i] * b[b_off + i];
        s1 += a[a_off + i + 1] * b[b_off + i + 1];
        s2 += a[a_off + i + 2] * b[b_off + i + 2];
        s3 += a[a_off + i + 3] * b[b_off + i + 3];
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < len; i++) sum += a[a_off + i] * b[b_off + i];
    return sum;
}

KOBLAS_KERNEL double koblas_dense_ssqd(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; i + 4 <= len; i += 4) {
        const double d0 = a[a_off + i] - b[b_off + i];
        const double d1 = a[a_off + i + 1] - b[b_off + i + 1];
        const double d2 = a[a_off + i + 2] - b[b_off + i + 2];
        const double d3 = a[a_off + i + 3] - b[b_off + i + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < len; i++) {
        const double d = a[a_off + i] - b[b_off + i];
        sum += d * d;
    }
    return sum;
}

static inline void koblas_dense_axpy_loop(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    for (int32_t i = 0; i < len; i++) y[y_off + i] += alpha * x[x_off + i];
}

KOBLAS_KERNEL void koblas_dense_axpy(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    if (alpha == 0.0) return;
    koblas_dense_axpy_loop(y, y_off, alpha, x, x_off, len);
}

KOBLAS_KERNEL void koblas_dense_axpy_arithmetic(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    koblas_dense_axpy_loop(y, y_off, alpha, x, x_off, len);
}

KOBLAS_KERNEL void koblas_dense_scale(double *v, int32_t v_off, double alpha, int32_t len) {
    if (alpha == 1.0) return;
    for (int32_t i = 0; i < len; i++) v[v_off + i] *= alpha;
}

KOBLAS_KERNEL double koblas_dense_nrm2(const double *v, int32_t v_off, int32_t len) {
    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;
    double q3 = 0.0;
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; i + 4 <= len; i += 4) {
        const double v0 = v[v_off + i];
        const double v1 = v[v_off + i + 1];
        const double v2 = v[v_off + i + 2];
        const double v3 = v[v_off + i + 3];
        q0 += v0 * v0;
        q1 += v1 * v1;
        q2 += v2 * v2;
        q3 += v3 * v3;
    }
    double squares = (q0 + q1) + (q2 + q3);
    for (; i < len; i++) {
        const double value = v[v_off + i];
        squares += value * value;
    }
    if (isfinite(squares) && squares >= 0x1p-1022) return sqrt(squares);

    double maximum = 0.0;
    for (int32_t k = 0; k < len; k++) {
        const double magnitude = fabs(v[v_off + k]);
        if (magnitude > maximum) maximum = magnitude;
    }
    if (maximum == 0.0 || isinf(maximum)) return sqrt(squares);

    double r0 = 0.0;
    double r1 = 0.0;
    double r2 = 0.0;
    double r3 = 0.0;
    int32_t j = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; j + 4 <= len; j += 4) {
        const double c0 = v[v_off + j] / maximum;
        const double c1 = v[v_off + j + 1] / maximum;
        const double c2 = v[v_off + j + 2] / maximum;
        const double c3 = v[v_off + j + 3] / maximum;
        r0 += c0 * c0;
        r1 += c1 * c1;
        r2 += c2 * c2;
        r3 += c3 * c3;
    }
    double scaled_squares = (r0 + r1) + (r2 + r3);
    for (; j < len; j++) {
        const double scaled = v[v_off + j] / maximum;
        scaled_squares += scaled * scaled;
    }
    return maximum * sqrt(scaled_squares);
}

KOBLAS_KERNEL double koblas_dense_sum(const double *v, int32_t v_off, int32_t len) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; i + 4 <= len; i += 4) {
        s0 += v[v_off + i];
        s1 += v[v_off + i + 1];
        s2 += v[v_off + i + 2];
        s3 += v[v_off + i + 3];
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < len; i++) sum += v[v_off + i];
    return sum;
}

KOBLAS_KERNEL double koblas_dense_asum(const double *v, int32_t v_off, int32_t len) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; i + 4 <= len; i += 4) {
        s0 += fabs(v[v_off + i]);
        s1 += fabs(v[v_off + i + 1]);
        s2 += fabs(v[v_off + i + 2]);
        s3 += fabs(v[v_off + i + 3]);
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; i < len; i++) sum += fabs(v[v_off + i]);
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

/* Strided on both operands, which may also overlap, so neither the loads nor the stores can pack. */
KOBLAS_KERNEL void koblas_dense_rotm(
    double *x, int32_t x_off, int32_t x_stride, double *y, int32_t y_off, int32_t y_stride,
    int32_t len, double h11, double h12, double h21, double h22
) {
    for (int32_t i = 0; i < len; i++) {
        const double xi = x[x_off + i * x_stride];
        const double yi = y[y_off + i * y_stride];
        x[x_off + i * x_stride] = h11 * xi + h12 * yi;
        y[y_off + i * y_stride] = h21 * xi + h22 * yi;
    }
}

/* Indexed loads cannot pack without a gather, but four chains still keep the adds off one another. */
KOBLAS_KERNEL double koblas_sparse_dot_dense(
    const int32_t *indices, const double *values, int32_t len, const double *dense
) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    int32_t k = 0;
    if (len >= KOBLAS_UNROLL_MIN) for (; k + 4 <= len; k += 4) {
        s0 += values[k] * dense[indices[k]];
        s1 += values[k + 1] * dense[indices[k + 1]];
        s2 += values[k + 2] * dense[indices[k + 2]];
        s3 += values[k + 3] * dense[indices[k + 3]];
    }
    double sum = (s0 + s1) + (s2 + s3);
    for (; k < len; k++) sum += values[k] * dense[indices[k]];
    return sum;
}

/* A merge, so the accumulate waits on the index comparison rather than on itself; one chain is enough. */
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
