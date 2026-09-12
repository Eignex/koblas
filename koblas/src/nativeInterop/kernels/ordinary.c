#include <math.h>
#include <stdint.h>
#include "internal.h"

#define koblas_dense_dot KOBLAS_IMPL(koblas_dense_dot)
#define koblas_dense_ssqd KOBLAS_IMPL(koblas_dense_ssqd)
#define koblas_dense_axpy KOBLAS_IMPL(koblas_dense_axpy)
#define koblas_dense_axpy_arithmetic KOBLAS_IMPL(koblas_dense_axpy_arithmetic)
#define koblas_dense_scale KOBLAS_IMPL(koblas_dense_scale)
#define koblas_dense_nrm2 KOBLAS_IMPL(koblas_dense_nrm2)
#define koblas_dense_sum KOBLAS_IMPL(koblas_dense_sum)
#define koblas_dense_asum KOBLAS_IMPL(koblas_dense_asum)
#define koblas_dense_iamax KOBLAS_IMPL(koblas_dense_iamax)
#define koblas_dense_gemm_tile KOBLAS_IMPL(koblas_dense_gemm_tile)
#define koblas_dense_swap KOBLAS_IMPL(koblas_dense_swap)
#define koblas_dense_dot4 KOBLAS_IMPL(koblas_dense_dot4)
#define koblas_dense_axpy4 KOBLAS_IMPL(koblas_dense_axpy4)
#define koblas_dense_dot_axpy KOBLAS_IMPL(koblas_dense_dot_axpy)
#define koblas_dense_rotm KOBLAS_IMPL(koblas_dense_rotm)
#define koblas_sparse_dot_dense KOBLAS_IMPL(koblas_sparse_dot_dense)
#define koblas_sparse_dot_sparse KOBLAS_IMPL(koblas_sparse_dot_sparse)
#define koblas_sparse_axpy KOBLAS_IMPL(koblas_sparse_axpy)
#define koblas_sparse_scatter KOBLAS_IMPL(koblas_sparse_scatter)
#define koblas_sparse_nrm2 KOBLAS_IMPL(koblas_sparse_nrm2)
#define koblas_sparse_gather KOBLAS_IMPL(koblas_sparse_gather)
#define koblas_sparse_gather_zero KOBLAS_IMPL(koblas_sparse_gather_zero)
#define koblas_dense_trsm_tile KOBLAS_IMPL(koblas_dense_trsm_tile)
#define koblas_dense_gemm_trsm_tile KOBLAS_IMPL(koblas_dense_gemm_trsm_tile)

/* Four independent accumulators expose reduction parallelism; short runs avoid their setup cost. */
#define KOBLAS_UNROLL_MIN 32

/* A logical group is four doubles, implemented by one AVX2 or two SSE2/NEON registers. */
#define KOBLAS_LANES 4
#define KOBLAS_VECTOR_STEP 16

typedef double koblas_v4d __attribute__((vector_size(KOBLAS_LANES * sizeof(double))));
typedef long long koblas_v4i __attribute__((vector_size(KOBLAS_LANES * sizeof(double))));

#define KOBLAS_LOAD(dst, ptr) __builtin_memcpy(&(dst), (ptr), sizeof(koblas_v4d))
#define KOBLAS_ZERO {0.0, 0.0, 0.0, 0.0}

/* Folds four lanes and four accumulators as a tree rather than a chain. */
#define KOBLAS_HORIZONTAL(t) (((t)[0] + (t)[1]) + ((t)[2] + (t)[3]))
#define KOBLAS_COMBINE(s0, s1, s2, s3) KOBLAS_HORIZONTAL((((s0) + (s1)) + ((s2) + (s3))))

/* The sparse dot keeps scalar chains: its loads are scattered, so there is no contiguous vector to form. */
#define KOBLAS_REPEAT_4(M) M(0) M(1) M(2) M(3)
#define KOBLAS_REPEAT(M) KOBLAS_REPEAT_4(M) M(4) M(5) M(6) M(7)
#define KOBLAS_ACCUMULATORS 8
#define KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7) \
    (((s0) + (s1)) + ((s2) + (s3))) + (((s4) + (s5)) + ((s6) + (s7)))

double koblas_dense_dot(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sum = 0.0; for (int32_t i = 0; i < len; i++) sum += a[a_off+i] * b[b_off+i]; return sum;
#else
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_VECTOR_STEP; i += KOBLAS_VECTOR_STEP) {
        koblas_v4d x0, x1, x2, x3, y0, y1, y2, y3;
        KOBLAS_LOAD(x0, a + a_off + i);
        KOBLAS_LOAD(y0, b + b_off + i);
        KOBLAS_LOAD(x1, a + a_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(y1, b + b_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(x2, a + a_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(y2, b + b_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(x3, a + a_off + i + 3 * KOBLAS_LANES);
        KOBLAS_LOAD(y3, b + b_off + i + 3 * KOBLAS_LANES);
        s0 += x0 * y0;
        s1 += x1 * y1;
        s2 += x2 * y2;
        s3 += x3 * y3;
    }
    double sum = KOBLAS_COMBINE(s0, s1, s2, s3);
    for (; i < len; i++) sum += a[a_off + i] * b[b_off + i];
    return sum;
#endif
}

double koblas_dense_ssqd(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sum = 0.0; for (int32_t i = 0; i < len; i++) { double d = a[a_off+i] - b[b_off+i]; sum += d*d; } return sum;
#else
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_VECTOR_STEP; i += KOBLAS_VECTOR_STEP) {
        koblas_v4d x0, x1, x2, x3, y0, y1, y2, y3;
        KOBLAS_LOAD(x0, a + a_off + i);
        KOBLAS_LOAD(y0, b + b_off + i);
        KOBLAS_LOAD(x1, a + a_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(y1, b + b_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(x2, a + a_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(y2, b + b_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(x3, a + a_off + i + 3 * KOBLAS_LANES);
        KOBLAS_LOAD(y3, b + b_off + i + 3 * KOBLAS_LANES);
        x0 -= y0;
        x1 -= y1;
        x2 -= y2;
        x3 -= y3;
        s0 += x0 * x0;
        s1 += x1 * x1;
        s2 += x2 * x2;
        s3 += x3 * x3;
    }
    double sum = KOBLAS_COMBINE(s0, s1, s2, s3);
    for (; i < len; i++) {
        const double d = a[a_off + i] - b[b_off + i];
        sum += d * d;
    }
    return sum;
#endif
}

static inline void koblas_dense_axpy_loop(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    for (int32_t i = 0; i < len; i++) y[y_off + i] += alpha * x[x_off + i];
}

void koblas_dense_axpy(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    if (alpha == 0.0) return;
    koblas_dense_axpy_loop(y, y_off, alpha, x, x_off, len);
}

void koblas_dense_axpy_arithmetic(
    double *y, int32_t y_off, double alpha, const double *x, int32_t x_off, int32_t len
) {
    koblas_dense_axpy_loop(y, y_off, alpha, x, x_off, len);
}

void koblas_dense_scale(double *v, int32_t v_off, double alpha, int32_t len) {
    if (alpha == 1.0 || len <= 0) return;
    v += v_off;
#if defined(__x86_64__) && !defined(KOBLAS_SCALAR_IMPL)
    // Cache-line-split AVX2 stores dominate medium vectors; shorter calls cannot amortize peeling.
    if (len >= 128) {
        while ((uintptr_t)v & (sizeof(koblas_v4d) - 1)) {
            *v++ *= alpha;
            --len;
        }
        double *aligned = __builtin_assume_aligned(v, sizeof(koblas_v4d));
        // Fixed pointer-relative blocks avoid scaled-index addresses in the vector stores.
        for (; len >= 16; len -= 16, aligned += 16) {
            for (int32_t i = 0; i < 16; i++) aligned[i] *= alpha;
        }
        for (int32_t i = 0; i < len; i++) aligned[i] *= alpha;
        return;
    }
#endif
    for (int32_t i = 0; i < len; i++) v[i] *= alpha;
}

double koblas_dense_nrm2(const double *v, int32_t v_off, int32_t len) {
#define KOBLAS_SQUARES_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SQUARES_STEP(q) \
    { const double value = v[v_off + i + q]; s##q += value * value; }
    KOBLAS_REPEAT(KOBLAS_SQUARES_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i <= len - KOBLAS_ACCUMULATORS; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SQUARES_STEP) }
    }
    double squares = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
#undef KOBLAS_SQUARES_DECLARE
#undef KOBLAS_SQUARES_STEP
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

#define KOBLAS_SCALED_DECLARE(q) double r##q = 0.0;
#define KOBLAS_SCALED_STEP(q) \
    { const double scaled = v[v_off + j + q] / maximum; r##q += scaled * scaled; }
    KOBLAS_REPEAT(KOBLAS_SCALED_DECLARE)
    int32_t j = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; j <= len - KOBLAS_ACCUMULATORS; j += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SCALED_STEP) }
    }
    double scaled_squares = KOBLAS_GATHER(r0, r1, r2, r3, r4, r5, r6, r7);
#undef KOBLAS_SCALED_DECLARE
#undef KOBLAS_SCALED_STEP
    for (; j < len; j++) {
        const double scaled = v[v_off + j] / maximum;
        scaled_squares += scaled * scaled;
    }
    return maximum * sqrt(scaled_squares);
}

double koblas_dense_sum(const double *v, int32_t v_off, int32_t len) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sum = 0.0; for (int32_t i = 0; i < len; i++) sum += v[v_off+i]; return sum;
#else
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_VECTOR_STEP; i += KOBLAS_VECTOR_STEP) {
        koblas_v4d x0, x1, x2, x3;
        KOBLAS_LOAD(x0, v + v_off + i);
        KOBLAS_LOAD(x1, v + v_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(x2, v + v_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(x3, v + v_off + i + 3 * KOBLAS_LANES);
        s0 += x0;
        s1 += x1;
        s2 += x2;
        s3 += x3;
    }
    double sum = KOBLAS_COMBINE(s0, s1, s2, s3);
    for (; i < len; i++) sum += v[v_off + i];
    return sum;
#endif
}

double koblas_dense_asum(const double *v, int32_t v_off, int32_t len) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sum = 0.0; for (int32_t i = 0; i < len; i++) sum += fabs(v[v_off+i]); return sum;
#else
    /* Clear the sign bit without changing NaN, infinity, or signed zero. */
    const koblas_v4i sign = {0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
                             0x7fffffffffffffffLL, 0x7fffffffffffffffLL};
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_VECTOR_STEP; i += KOBLAS_VECTOR_STEP) {
        koblas_v4d x0, x1, x2, x3;
        koblas_v4i b0, b1, b2, b3;
        KOBLAS_LOAD(x0, v + v_off + i);
        KOBLAS_LOAD(x1, v + v_off + i + KOBLAS_LANES);
        KOBLAS_LOAD(x2, v + v_off + i + 2 * KOBLAS_LANES);
        KOBLAS_LOAD(x3, v + v_off + i + 3 * KOBLAS_LANES);
        __builtin_memcpy(&b0, &x0, sizeof b0);
        __builtin_memcpy(&b1, &x1, sizeof b1);
        __builtin_memcpy(&b2, &x2, sizeof b2);
        __builtin_memcpy(&b3, &x3, sizeof b3);
        b0 &= sign;
        b1 &= sign;
        b2 &= sign;
        b3 &= sign;
        __builtin_memcpy(&x0, &b0, sizeof x0);
        __builtin_memcpy(&x1, &b1, sizeof x1);
        __builtin_memcpy(&x2, &b2, sizeof x2);
        __builtin_memcpy(&x3, &b3, sizeof x3);
        s0 += x0;
        s1 += x1;
        s2 += x2;
        s3 += x3;
    }
    double sum = KOBLAS_COMBINE(s0, s1, s2, s3);
    for (; i < len; i++) sum += fabs(v[v_off + i]);
    return sum;
#endif
}

/* Strict lane comparisons ignore NaNs; the second pass resolves ties in input order. */
int32_t koblas_dense_iamax(const double *v, int32_t v_off, int32_t len) {
#if defined(KOBLAS_SCALAR_IMPL)
    if (len == 0) return -1; double best = 0.0; int32_t index = 0; for (int32_t i = 0; i < len; i++) { double a = fabs(v[v_off+i]); if (a > best) { best = a; index = i; } } return index;
#else

    if (len == 0) return -1;
    const koblas_v4i sign = {0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
                             0x7fffffffffffffffLL, 0x7fffffffffffffffLL};
    /* Named accumulators keep GCC from spilling a vector array on every iteration. */
    koblas_v4d m0 = KOBLAS_ZERO, m1 = KOBLAS_ZERO, m2 = KOBLAS_ZERO, m3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_VECTOR_STEP; i += KOBLAS_VECTOR_STEP) {
        {
            koblas_v4i bits, previous;
            koblas_v4d values;
            KOBLAS_LOAD(values, v + v_off + i);
            __builtin_memcpy(&bits, &values, sizeof bits);
            bits &= sign;
            __builtin_memcpy(&values, &bits, sizeof values);
            koblas_v4i greater = values > m0;
            __builtin_memcpy(&previous, &m0, sizeof previous);
            bits = (bits & greater) | (previous & ~greater);
            __builtin_memcpy(&m0, &bits, sizeof bits);
        }
        {
            koblas_v4i bits, previous;
            koblas_v4d values;
            KOBLAS_LOAD(values, v + v_off + i + KOBLAS_LANES);
            __builtin_memcpy(&bits, &values, sizeof bits);
            bits &= sign;
            __builtin_memcpy(&values, &bits, sizeof values);
            koblas_v4i greater = values > m1;
            __builtin_memcpy(&previous, &m1, sizeof previous);
            bits = (bits & greater) | (previous & ~greater);
            __builtin_memcpy(&m1, &bits, sizeof bits);
        }
        {
            koblas_v4i bits, previous;
            koblas_v4d values;
            KOBLAS_LOAD(values, v + v_off + i + 2 * KOBLAS_LANES);
            __builtin_memcpy(&bits, &values, sizeof bits);
            bits &= sign;
            __builtin_memcpy(&values, &bits, sizeof values);
            koblas_v4i greater = values > m2;
            __builtin_memcpy(&previous, &m2, sizeof previous);
            bits = (bits & greater) | (previous & ~greater);
            __builtin_memcpy(&m2, &bits, sizeof bits);
        }
        {
            koblas_v4i bits, previous;
            koblas_v4d values;
            KOBLAS_LOAD(values, v + v_off + i + 3 * KOBLAS_LANES);
            __builtin_memcpy(&bits, &values, sizeof bits);
            bits &= sign;
            __builtin_memcpy(&values, &bits, sizeof values);
            koblas_v4i greater = values > m3;
            __builtin_memcpy(&previous, &m3, sizeof previous);
            bits = (bits & greater) | (previous & ~greater);
            __builtin_memcpy(&m3, &bits, sizeof bits);
        }
    }
    double best = 0.0;
    for (int lane = 0; lane < KOBLAS_LANES; lane++) {
        if (m0[lane] > best) best = m0[lane];
        if (m1[lane] > best) best = m1[lane];
        if (m2[lane] > best) best = m2[lane];
        if (m3[lane] > best) best = m3[lane];
    }
    for (; i < len; i++) {
        double magnitude = fabs(v[v_off + i]);
        if (magnitude > best) best = magnitude;
    }
    if (best == 0.0) return 0;
    const koblas_v4d target = {best, best, best, best};
    for (i = 0; i <= len - KOBLAS_LANES; i += KOBLAS_LANES) {
        koblas_v4d values;
        koblas_v4i bits;
        KOBLAS_LOAD(values, v + v_off + i);
        __builtin_memcpy(&bits, &values, sizeof bits);
        bits &= sign;
        __builtin_memcpy(&values, &bits, sizeof values);
        koblas_v4i matches = values == target;
        if (matches[0]) return i;
        if (matches[1]) return i + 1;
        if (matches[2]) return i + 2;
        if (matches[3]) return i + 3;
    }
    for (; i < len; i++) {
        if (fabs(v[v_off + i]) == best) return i;
    }
    return 0;
#endif
}

/* Four-row depth-major groups feed sixteen independent FP64 accumulation chains. The same logical
 * tile uses one AVX2 or two SSE2/NEON registers per output column; packing does not imply register width.
 * Each tile accumulates its whole depth before writing C, including when C aliases a packed input.
 */
#define KOBLAS_GEMM_TILE 4

void koblas_dense_gemm_tile(
    int32_t depth,
    const double *packed_a, int32_t a_off,
    const double *packed_b, int32_t b_off,
    double *c, int32_t c_off, int32_t ldc
) {
    double c00 = 0.0, c10 = 0.0, c20 = 0.0, c30 = 0.0;
    double c01 = 0.0, c11 = 0.0, c21 = 0.0, c31 = 0.0;
    double c02 = 0.0, c12 = 0.0, c22 = 0.0, c32 = 0.0;
    double c03 = 0.0, c13 = 0.0, c23 = 0.0, c33 = 0.0;
    const double *ap = packed_a + a_off;
    const double *bp = packed_b + b_off;
    for (int32_t p = 0; p < depth; p++) {
        const double a0 = ap[0];
        const double a1 = ap[1];
        const double a2 = ap[2];
        const double a3 = ap[3];
        double coefficient = bp[0];
        c00 += a0 * coefficient;
        c10 += a1 * coefficient;
        c20 += a2 * coefficient;
        c30 += a3 * coefficient;
        coefficient = bp[1];
        c01 += a0 * coefficient;
        c11 += a1 * coefficient;
        c21 += a2 * coefficient;
        c31 += a3 * coefficient;
        coefficient = bp[2];
        c02 += a0 * coefficient;
        c12 += a1 * coefficient;
        c22 += a2 * coefficient;
        c32 += a3 * coefficient;
        coefficient = bp[3];
        c03 += a0 * coefficient;
        c13 += a1 * coefficient;
        c23 += a2 * coefficient;
        c33 += a3 * coefficient;
        ap += KOBLAS_GEMM_TILE;
        bp += KOBLAS_GEMM_TILE;
    }
    double *out = c + c_off;
    out[0] += c00;
    out[1] += c10;
    out[2] += c20;
    out[3] += c30;
    out[ldc] += c01;
    out[ldc + 1] += c11;
    out[ldc + 2] += c21;
    out[ldc + 3] += c31;
    out[2 * ldc] += c02;
    out[2 * ldc + 1] += c12;
    out[2 * ldc + 2] += c22;
    out[2 * ldc + 3] += c32;
    out[3 * ldc] += c03;
    out[3 * ldc + 1] += c13;
    out[3 * ldc + 2] += c23;
    out[3 * ldc + 3] += c33;
}

void koblas_dense_swap(
    double *a, int32_t a_off, double *b, int32_t b_off, int32_t len
) {
    for (int32_t i = 0; i < len; i++) {
        const double temporary = a[a_off + i];
        a[a_off + i] = b[b_off + i];
        b[b_off + i] = temporary;
    }
}

void koblas_dense_dot4(
    const double *a, int32_t a_off, int32_t stride, const double *b, int32_t b_off,
    int32_t len, double *out, int32_t out_off
) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sums[4] = {0.0, 0.0, 0.0, 0.0}; for (int32_t i = 0; i < len; i++) { double x = b[b_off+i]; for (int32_t j = 0; j < 4; j++) sums[j] += a[a_off+j*stride+i]*x; } for (int32_t j = 0; j < 4; j++) out[out_off+j] = sums[j];
#else

    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    const double *r0 = a + a_off;
    const double *r1 = r0 + stride;
    const double *r2 = r1 + stride;
    const double *r3 = r2 + stride;
    int32_t i = 0;
    for (; i <= len - KOBLAS_LANES; i += KOBLAS_LANES) {
        koblas_v4d shared, x0, x1, x2, x3;
        KOBLAS_LOAD(shared, b + b_off + i);
        KOBLAS_LOAD(x0, r0 + i);
        KOBLAS_LOAD(x1, r1 + i);
        KOBLAS_LOAD(x2, r2 + i);
        KOBLAS_LOAD(x3, r3 + i);
        s0 += x0 * shared;
        s1 += x1 * shared;
        s2 += x2 * shared;
        s3 += x3 * shared;
    }
    double t0 = KOBLAS_HORIZONTAL(s0);
    double t1 = KOBLAS_HORIZONTAL(s1);
    double t2 = KOBLAS_HORIZONTAL(s2);
    double t3 = KOBLAS_HORIZONTAL(s3);
    for (; i < len; i++) {
        const double shared = b[b_off + i];
        t0 += r0[i] * shared;
        t1 += r1[i] * shared;
        t2 += r2[i] * shared;
        t3 += r3[i] * shared;
    }
    out[out_off] = t0;
    out[out_off + 1] = t1;
    out[out_off + 2] = t2;
    out[out_off + 3] = t3;
#endif
}

void koblas_dense_axpy4(
    double *y, int32_t y_off, const double *a, int32_t a_off, int32_t stride,
    double c0, double c1, double c2, double c3, int32_t len
) {
#if defined(KOBLAS_SCALAR_IMPL)
    for (int32_t i = 0; i < len; i++) { double value = y[y_off+i]; value += c0*a[a_off+i]; value += c1*a[a_off+stride+i]; value += c2*a[a_off+2*stride+i]; value += c3*a[a_off+3*stride+i]; y[y_off+i] = value; }
#else

    const double *r0 = a + a_off;
    const double *r1 = r0 + stride;
    const double *r2 = r1 + stride;
    const double *r3 = r2 + stride;
    int32_t i = 0;
    for (; i <= len - KOBLAS_LANES; i += KOBLAS_LANES) {
        koblas_v4d value, a0, a1, a2, a3;
        KOBLAS_LOAD(value, y + y_off + i);
        KOBLAS_LOAD(a0, r0 + i);
        KOBLAS_LOAD(a1, r1 + i);
        KOBLAS_LOAD(a2, r2 + i);
        KOBLAS_LOAD(a3, r3 + i);
        value += c0 * a0;
        value += c1 * a1;
        value += c2 * a2;
        value += c3 * a3;
        __builtin_memcpy(y + y_off + i, &value, sizeof value);
    }
    for (; i < len; i++) {
        double value = y[y_off + i];
        value += c0 * r0[i];
        value += c1 * r1[i];
        value += c2 * r2[i];
        value += c3 * r3[i];
        y[y_off + i] = value;
    }
#endif
}

double koblas_dense_dot_axpy(
    double *y, int32_t y_off, double alpha, const double *a, int32_t a_off,
    const double *x, int32_t x_off, int32_t len
) {
#if defined(KOBLAS_SCALAR_IMPL)
    double sum = 0.0; for (int32_t i = 0; i < len; i++) { double av = a[a_off+i], xv = x[x_off+i]; sum += av*xv; y[y_off+i] += alpha*av; } return sum;
#else

    koblas_v4d sum = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i <= len - KOBLAS_LANES; i += KOBLAS_LANES) {
        koblas_v4d av, xv, yv;
        KOBLAS_LOAD(av, a + a_off + i);
        KOBLAS_LOAD(xv, x + x_off + i);
        KOBLAS_LOAD(yv, y + y_off + i);
        sum += av * xv;
        yv += alpha * av;
        __builtin_memcpy(y + y_off + i, &yv, sizeof yv);
    }
    double result = KOBLAS_HORIZONTAL(sum);
    for (; i < len; i++) {
        const double ai = a[a_off + i];
        const double xi = x[x_off + i];
        result += ai * xi;
        y[y_off + i] += alpha * ai;
    }
    return result;
#endif
}

/* Strided on both operands, which may also overlap, so neither the loads nor the stores can pack. */
void koblas_dense_rotm(
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

#if defined(KOBLAS_SCALAR_IMPL)
/* Indexed loads cannot pack without a gather, but four chains still keep the adds off one another. */
double koblas_sparse_dot_dense(
    const int32_t *indices, int32_t index_off,
    const double *values, int32_t value_off, int32_t len, const double *dense
) {
#define KOBLAS_SPARSE_DOT_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SPARSE_DOT_STEP(q) \
    s##q += values[value_off + k + q] * dense[indices[index_off + k + q]];
    KOBLAS_REPEAT(KOBLAS_SPARSE_DOT_DECLARE)
    int32_t k = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; k <= len - KOBLAS_ACCUMULATORS; k += KOBLAS_ACCUMULATORS) {
            KOBLAS_REPEAT(KOBLAS_SPARSE_DOT_STEP)
        }
    }
    double sum = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
#undef KOBLAS_SPARSE_DOT_DECLARE
#undef KOBLAS_SPARSE_DOT_STEP
    for (; k < len; k++) sum += values[value_off + k] * dense[indices[index_off + k]];
    return sum;
}

/* A merge, so the accumulate waits on the index comparison rather than on itself; one chain is enough. */
double koblas_sparse_dot_sparse(
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

void koblas_sparse_axpy(
    const int32_t *indices, int32_t index_off,
    const double *values, int32_t value_off, int32_t len, double alpha, double *dense
) {
    if (alpha == 0.0) return;
    for (int32_t k = 0; k < len; k++) {
        /* Preserve portable gemv overflow semantics by preventing multiply-add contraction. */
        volatile double increment = alpha * values[value_off + k];
        dense[indices[index_off + k]] += increment;
    }
}

void koblas_sparse_scatter(
    const int32_t *indices, int32_t index_off,
    const double *values, int32_t value_off, int32_t len, double *dense
) {
    for (int32_t k = 0; k < len; k++) dense[indices[index_off + k]] = values[value_off + k];
}

double koblas_sparse_nrm2(
    const int32_t *indices, int32_t index_off, int32_t len, const double *values
) {
    double squares = 0.0;
    for (int32_t k = 0; k < len; k++) {
        const double value = values[indices[index_off + k]];
        squares += value * value;
    }
    if (isfinite(squares) && squares >= 0x1p-1022) return sqrt(squares);

    double maximum = 0.0;
    for (int32_t k = 0; k < len; k++) {
        const double magnitude = fabs(values[indices[index_off + k]]);
        if (magnitude > maximum) maximum = magnitude;
    }
    if (maximum == 0.0 || isinf(maximum)) return sqrt(squares);

    double scaled_squares = 0.0;
    for (int32_t k = 0; k < len; k++) {
        const double scaled = values[indices[index_off + k]] / maximum;
        scaled_squares += scaled * scaled;
    }
    return maximum * sqrt(scaled_squares);
}

void koblas_sparse_gather(
    const int32_t *indices, double *values, int32_t len, const double *dense
) {
    for (int32_t k = 0; k < len; k++) values[k] = dense[indices[k]];
}

void koblas_sparse_gather_zero(
    const int32_t *indices, double *values, int32_t len, double *dense
) {
    for (int32_t k = 0; k < len; k++) {
        const int32_t index = indices[k];
        values[k] = dense[index];
        dense[index] = 0.0;
    }
}

#endif

/*
 * Packed right-side triangular solve. T uses the packed-right row groups consumed by the GEMM tile, while
 * X uses its column-major output layout. Exact-zero coefficients are skipped so structural zeroes never
 * form products with non-finite solved values.
 */
void koblas_dense_trsm_tile(
    int32_t valid_rows, int32_t order,
    const double *packed_triangle, int32_t triangle_off,
    int32_t lower, int32_t unit_diag,
    double *x, int32_t x_off
) {
    const double *triangle = packed_triangle + triangle_off;
    double *out = x + x_off;
    if (lower) {
        for (int32_t j = order - 1; j >= 0; j--) {
            double *solved = out + j * KOBLAS_GEMM_TILE;
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < valid_rows; row++) solved[row] /= diagonal;
            }
            for (int32_t column = 0; column < j; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    double *target = out + column * KOBLAS_GEMM_TILE;
                    for (int32_t row = 0; row < valid_rows; row++) {
                        target[row] -= solved[row] * coefficient;
                    }
                }
            }
        }
    } else {
        for (int32_t j = 0; j < order; j++) {
            double *solved = out + j * KOBLAS_GEMM_TILE;
            if (!unit_diag) {
                const double diagonal = triangle[j * KOBLAS_GEMM_TILE + j];
                for (int32_t row = 0; row < valid_rows; row++) solved[row] /= diagonal;
            }
            for (int32_t column = j + 1; column < order; column++) {
                const double coefficient = triangle[j * KOBLAS_GEMM_TILE + column];
                if (coefficient != 0.0) {
                    double *target = out + column * KOBLAS_GEMM_TILE;
                    for (int32_t row = 0; row < valid_rows; row++) {
                        target[row] -= solved[row] * coefficient;
                    }
                }
            }
        }
    }
}

/*
 * Product intermediates stay in scalar locals for full and edge tiles. Packed product groups are always padded;
 * an edge initializes and writes only its logical X entries before handing that exact-size tile to the solve.
 */
void koblas_dense_gemm_trsm_tile(
    int32_t depth, int32_t valid_rows, int32_t order,
    const double *packed_a, int32_t a_off,
    const double *packed_b, int32_t b_off,
    const double *packed_triangle, int32_t triangle_off,
    int32_t lower, int32_t unit_diag,
    double *x, int32_t x_off
) {
    double *out = x + x_off;
    double c00 = valid_rows > 0 && order > 0 ? out[0] : 0.0;
    double c10 = valid_rows > 1 && order > 0 ? out[1] : 0.0;
    double c20 = valid_rows > 2 && order > 0 ? out[2] : 0.0;
    double c30 = valid_rows > 3 && order > 0 ? out[3] : 0.0;
    double c01 = valid_rows > 0 && order > 1 ? out[4] : 0.0;
    double c11 = valid_rows > 1 && order > 1 ? out[5] : 0.0;
    double c21 = valid_rows > 2 && order > 1 ? out[6] : 0.0;
    double c31 = valid_rows > 3 && order > 1 ? out[7] : 0.0;
    double c02 = valid_rows > 0 && order > 2 ? out[8] : 0.0;
    double c12 = valid_rows > 1 && order > 2 ? out[9] : 0.0;
    double c22 = valid_rows > 2 && order > 2 ? out[10] : 0.0;
    double c32 = valid_rows > 3 && order > 2 ? out[11] : 0.0;
    double c03 = valid_rows > 0 && order > 3 ? out[12] : 0.0;
    double c13 = valid_rows > 1 && order > 3 ? out[13] : 0.0;
    double c23 = valid_rows > 2 && order > 3 ? out[14] : 0.0;
    double c33 = valid_rows > 3 && order > 3 ? out[15] : 0.0;
    const double *ap = packed_a + a_off;
    const double *bp = packed_b + b_off;
    for (int32_t p = 0; p < depth; p++) {
        const double a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
        double coefficient = bp[0];
        c00 -= a0 * coefficient; c10 -= a1 * coefficient;
        c20 -= a2 * coefficient; c30 -= a3 * coefficient;
        coefficient = bp[1];
        c01 -= a0 * coefficient; c11 -= a1 * coefficient;
        c21 -= a2 * coefficient; c31 -= a3 * coefficient;
        coefficient = bp[2];
        c02 -= a0 * coefficient; c12 -= a1 * coefficient;
        c22 -= a2 * coefficient; c32 -= a3 * coefficient;
        coefficient = bp[3];
        c03 -= a0 * coefficient; c13 -= a1 * coefficient;
        c23 -= a2 * coefficient; c33 -= a3 * coefficient;
        ap += KOBLAS_GEMM_TILE;
        bp += KOBLAS_GEMM_TILE;
    }
    if (valid_rows != KOBLAS_GEMM_TILE || order != KOBLAS_GEMM_TILE) {
        if (order > 0) {
            if (valid_rows > 0) out[0] = c00;
            if (valid_rows > 1) out[1] = c10;
            if (valid_rows > 2) out[2] = c20;
            if (valid_rows > 3) out[3] = c30;
        }
        if (order > 1) {
            if (valid_rows > 0) out[4] = c01;
            if (valid_rows > 1) out[5] = c11;
            if (valid_rows > 2) out[6] = c21;
            if (valid_rows > 3) out[7] = c31;
        }
        if (order > 2) {
            if (valid_rows > 0) out[8] = c02;
            if (valid_rows > 1) out[9] = c12;
            if (valid_rows > 2) out[10] = c22;
            if (valid_rows > 3) out[11] = c32;
        }
        if (order > 3) {
            if (valid_rows > 0) out[12] = c03;
            if (valid_rows > 1) out[13] = c13;
            if (valid_rows > 2) out[14] = c23;
            if (valid_rows > 3) out[15] = c33;
        }
        koblas_dense_trsm_tile(
            valid_rows, order, packed_triangle, triangle_off, lower, unit_diag, x, x_off
        );
        return;
    }
    const double *triangle = packed_triangle + triangle_off;
#define KOBLAS_DIVIDE_COLUMN(a, b, c, d, diagonal) \
    do { a /= diagonal; b /= diagonal; c /= diagonal; d /= diagonal; } while (0)
#define KOBLAS_SUBTRACT_COLUMN(a, b, c, d, sa, sb, sc, sd, coefficient) \
    do { \
        a -= sa * coefficient; b -= sb * coefficient; \
        c -= sc * coefficient; d -= sd * coefficient; \
    } while (0)
    if (lower) {
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c03, c13, c23, c33, triangle[15]);
        double coefficient = triangle[12];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c03, c13, c23, c33, coefficient);
        coefficient = triangle[13];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c03, c13, c23, c33, coefficient);
        coefficient = triangle[14];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c03, c13, c23, c33, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c02, c12, c22, c32, triangle[10]);
        coefficient = triangle[8];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c02, c12, c22, c32, coefficient);
        coefficient = triangle[9];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c02, c12, c22, c32, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c01, c11, c21, c31, triangle[5]);
        coefficient = triangle[4];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c00, c10, c20, c30, c01, c11, c21, c31, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c00, c10, c20, c30, triangle[0]);
    } else {
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c00, c10, c20, c30, triangle[0]);
        double coefficient = triangle[1];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c01, c11, c21, c31, c00, c10, c20, c30, coefficient);
        coefficient = triangle[2];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c00, c10, c20, c30, coefficient);
        coefficient = triangle[3];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c00, c10, c20, c30, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c01, c11, c21, c31, triangle[5]);
        coefficient = triangle[6];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c02, c12, c22, c32, c01, c11, c21, c31, coefficient);
        coefficient = triangle[7];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c01, c11, c21, c31, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c02, c12, c22, c32, triangle[10]);
        coefficient = triangle[11];
        if (coefficient != 0.0) KOBLAS_SUBTRACT_COLUMN(c03, c13, c23, c33, c02, c12, c22, c32, coefficient);
        if (!unit_diag) KOBLAS_DIVIDE_COLUMN(c03, c13, c23, c33, triangle[15]);
    }
#undef KOBLAS_SUBTRACT_COLUMN
#undef KOBLAS_DIVIDE_COLUMN
    out[0] = c00; out[1] = c10; out[2] = c20; out[3] = c30;
    out[4] = c01; out[5] = c11; out[6] = c21; out[7] = c31;
    out[8] = c02; out[9] = c12; out[10] = c22; out[11] = c32;
    out[12] = c03; out[13] = c13; out[14] = c23; out[15] = c33;
}
