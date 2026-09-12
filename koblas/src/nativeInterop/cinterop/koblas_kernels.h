#ifndef KOBLAS_KERNELS_H
#define KOBLAS_KERNELS_H

#include <math.h>
#include <stdint.h>

/* x86-64 ELF dispatches between baseline and AVX2 clones; other targets use the baseline implementation. */
#if defined(KOBLAS_KERNELS_IMPLEMENTATION)
/*
 * Exported through a pragma rather than an attribute per function, because clang refuses to accept a
 * visibility attribute on the same declaration as target_clones and Kotlin/Native compiles this header
 * with clang. The pragma is honoured by both compilers, survives the hidden default the shared library is
 * built with, and leaves the clone attribute alone. It is popped at the end of the header.
 */
#pragma GCC visibility push(default)
#if defined(__x86_64__) && defined(__ELF__) && \
    ((defined(__clang__) && __clang_major__ >= 14) || \
     (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 6))
#define KOBLAS_KERNEL __attribute__((target_clones("avx2", "default")))
#else
#define KOBLAS_KERNEL
#endif
#define KOBLAS_KERNEL_BASELINE
#else
#define KOBLAS_KERNEL static inline
#define KOBLAS_KERNEL_BASELINE static inline
#endif

/* Four independent accumulators expose reduction parallelism; short runs avoid their setup cost. */
#define KOBLAS_UNROLL_MIN 32

/* Explicit vectors guarantee lane width. Keep them within a function body for clang's target-clone ABI. */
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

KOBLAS_KERNEL double koblas_dense_dot(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i + KOBLAS_VECTOR_STEP <= len; i += KOBLAS_VECTOR_STEP) {
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
}

KOBLAS_KERNEL double koblas_dense_ssqd(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i + KOBLAS_VECTOR_STEP <= len; i += KOBLAS_VECTOR_STEP) {
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
#define KOBLAS_SQUARES_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SQUARES_STEP(q) \
    { const double value = v[v_off + i + q]; s##q += value * value; }
    KOBLAS_REPEAT(KOBLAS_SQUARES_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i + KOBLAS_ACCUMULATORS <= len; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SQUARES_STEP) }
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
        for (; j + KOBLAS_ACCUMULATORS <= len; j += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SCALED_STEP) }
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

KOBLAS_KERNEL double koblas_dense_sum(const double *v, int32_t v_off, int32_t len) {    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i + KOBLAS_VECTOR_STEP <= len; i += KOBLAS_VECTOR_STEP) {
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
}

KOBLAS_KERNEL double koblas_dense_asum(const double *v, int32_t v_off, int32_t len) {    /* Clear the sign bit without changing NaN, infinity, or signed zero. */
    const koblas_v4i sign = {0x7fffffffffffffffLL, 0x7fffffffffffffffLL,
                             0x7fffffffffffffffLL, 0x7fffffffffffffffLL};
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i + KOBLAS_VECTOR_STEP <= len; i += KOBLAS_VECTOR_STEP) {
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
}

/* Strict lane comparisons ignore NaNs; the second pass resolves ties in input order. */
KOBLAS_KERNEL int32_t koblas_dense_iamax(const double *v, int32_t v_off, int32_t len) {
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
}

/*
 * The matrix-product tile. koblas packs both operands into panels laid out in the order read here, and this
 * accumulates KOBLAS_GEMM_TILE rows by KOBLAS_GEMM_TILE columns of C over the whole of depth before touching
 * C at all, which is the point: C is read and written once per tile rather than once per step.
 *
 * Sixteen accumulators in named locals, four rows by four columns. Four rows is two SSE2 registers or one
 * AVX2 register, so the tile occupies eight vector registers at the baseline and four when a wider clone
 * runs, either of which leaves room for the operand loads. A wider tile would hold more of C per pass and
 * spill at the baseline, which is the trade this shape settles on the narrow side because the same source
 * has to compile well both ways.
 *
 * One call covers depth times sixteen multiply-adds, so the cost of reaching it from a managed caller is
 * spread over thousands of operations. That is what makes a tile the right unit to put behind a foreign
 * call, where a single vector operation is not.
 */
#define KOBLAS_GEMM_TILE 4

KOBLAS_KERNEL void koblas_dense_gemm_tile(
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
    koblas_v4d s0 = KOBLAS_ZERO, s1 = KOBLAS_ZERO, s2 = KOBLAS_ZERO, s3 = KOBLAS_ZERO;
    const double *r0 = a + a_off;
    const double *r1 = r0 + stride;
    const double *r2 = r1 + stride;
    const double *r3 = r2 + stride;
    int32_t i = 0;
    for (; i + KOBLAS_LANES <= len; i += KOBLAS_LANES) {
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
}

KOBLAS_KERNEL void koblas_dense_axpy4(
    double *y, int32_t y_off, const double *a, int32_t a_off, int32_t stride,
    double c0, double c1, double c2, double c3, int32_t len
) {
    const double *r0 = a + a_off;
    const double *r1 = r0 + stride;
    const double *r2 = r1 + stride;
    const double *r3 = r2 + stride;
    int32_t i = 0;
    for (; i + KOBLAS_LANES <= len; i += KOBLAS_LANES) {
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
}

KOBLAS_KERNEL double koblas_dense_dot_axpy(
    double *y, int32_t y_off, double alpha, const double *a, int32_t a_off,
    const double *x, int32_t x_off, int32_t len
) {
    koblas_v4d sum = KOBLAS_ZERO;
    int32_t i = 0;
    for (; i + KOBLAS_LANES <= len; i += KOBLAS_LANES) {
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
    const int32_t *indices, int32_t index_off,
    const double *values, int32_t value_off, int32_t len, const double *dense
) {
#define KOBLAS_SPARSE_DOT_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SPARSE_DOT_STEP(q) \
    s##q += values[value_off + k + q] * dense[indices[index_off + k + q]];
    KOBLAS_REPEAT(KOBLAS_SPARSE_DOT_DECLARE)
    int32_t k = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; k + KOBLAS_ACCUMULATORS <= len; k += KOBLAS_ACCUMULATORS) {
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

KOBLAS_KERNEL void koblas_sparse_scatter(
    const int32_t *indices, int32_t index_off,
    const double *values, int32_t value_off, int32_t len, double *dense
) {
    for (int32_t k = 0; k < len; k++) dense[indices[index_off + k]] = values[value_off + k];
}

KOBLAS_KERNEL double koblas_sparse_nrm2(
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

#include "koblas_packed_trsm.h"

#undef KOBLAS_KERNEL
#if defined(KOBLAS_KERNELS_IMPLEMENTATION)
#pragma GCC visibility pop
#endif

#endif
