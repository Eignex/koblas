#ifndef KOBLAS_KERNELS_H
#define KOBLAS_KERNELS_H

#include <math.h>
#include <stdint.h>

/*
 * The shared library the JVM loads is built once and shipped to unknown machines, so it cannot be compiled
 * for a wide instruction set outright. Without a target flag the compiler assumes only the x86-64 baseline,
 * which is SSE2, and every loop here vectorises two doubles at a time while the JIT it competes against
 * emits four. target_clones resolves that: the compiler emits a baseline version and an AVX2 version of
 * each kernel and picks between them once, through an ifunc resolver, the first time the symbol is called.
 * A machine without AVX2 runs exactly what it ran before.
 *
 * Both clones return the same bits. The accumulator structure below is written out in the source, so which
 * additions are grouped together is fixed there rather than chosen by the vectoriser, and a wider register
 * holds the same four accumulators in one place instead of two. A result therefore does not depend on which
 * clone the machine resolved to.
 *
 * KOBLAS_KERNEL_BASELINE opts a kernel out. Only koblas_dense_dot4 uses it: timed against its own baseline
 * at 512 and 2048 elements the AVX2 clone runs 1.7 times slower, the one kernel here that the wider
 * registers hurt. It carries five live streams, four strided rows against one shared operand, and at twice
 * the register width they no longer fit. Every other kernel gains or is unchanged, so the exception is
 * per-kernel rather than a reason to drop the clones.
 *
 * Only x86-64 ELF takes the clones. Aarch64 has no equivalent split, since NEON is baseline there and the
 * next step up is SVE, which needs different code rather than the same code widened. Mach-O has no ifunc.
 * Kotlin/Native compiles this header as static inline through its own toolchain rather than linking the
 * shared library, so it stays on the baseline: widening it would mean pinning the published Native
 * artifacts to a newer instruction set than they target today.
 */
#if defined(KOBLAS_KERNELS_IMPLEMENTATION)
#if defined(__x86_64__) && defined(__ELF__) && \
    ((defined(__clang__) && __clang_major__ >= 14) || \
     (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 6))
#define KOBLAS_KERNEL \
    __attribute__((visibility("default"))) __attribute__((target_clones("avx2", "default")))
#else
#define KOBLAS_KERNEL __attribute__((visibility("default")))
#endif
#define KOBLAS_KERNEL_BASELINE __attribute__((visibility("default")))
#else
#define KOBLAS_KERNEL static inline
#define KOBLAS_KERNEL_BASELINE static inline
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

/*
 * How many independent accumulator chains a reduction carries, and the machinery that writes them out.
 *
 * The chains have to be named locals. An array of them, even fully unrolled with constant indices and an
 * unroll pragma, stays in memory: measured on a dot at length 1024 and 4096 the array form takes 1362 to
 * 5503 ns where the named form takes 74 to 425, so it is ten to thirteen times worse. Rather than hand
 * write eight chains in each of five kernels, KOBLAS_REPEAT applies a macro to each index and each kernel
 * supplies the three bodies it needs: one to declare a chain, one to advance it, one to fold it in.
 *
 * Eight rather than four, which is what these carried before: on a dot at 1024 eight takes 74 ns against
 * 125, and on an absolute sum at 4096 233 against 387. Sixteen is worse on AVX2 at both lengths and only
 * marginally better at the baseline, so it does not pay for a second shape.
 *
 * The count is one number rather than one derived from the vector width, and it has to be. KOBLAS_KERNEL
 * asks the compiler for a baseline clone and an AVX2 clone of the same preprocessed source, so __AVX2__
 * and __AVX512F__ are not defined while either is generated and a width-derived count would silently
 * resolve to the baseline in both. Eight is the value that measured best under each clone. Per-clone counts
 * would need the kernels compiled once per instruction set behind a runtime resolver, which is a different
 * build from this one.
 */
#define KOBLAS_ACCUMULATORS 8

#define KOBLAS_REPEAT_4(M) M(0) M(1) M(2) M(3)
#define KOBLAS_REPEAT(M) KOBLAS_REPEAT_4(M) M(4) M(5) M(6) M(7)

/* Folds the chains in pairs rather than in sequence, so the combine is a tree and not a dependency chain. */
#define KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7) \
    (((s0) + (s1)) + ((s2) + (s3))) + (((s4) + (s5)) + ((s6) + (s7)))

KOBLAS_KERNEL double koblas_dense_dot(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
#define KOBLAS_DOT_DECLARE(q) double s##q = 0.0;
#define KOBLAS_DOT_STEP(q) s##q += a[a_off + i + q] * b[b_off + i + q];
    KOBLAS_REPEAT(KOBLAS_DOT_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i + KOBLAS_ACCUMULATORS <= len; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_DOT_STEP) }
    }
    double sum = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
    for (; i < len; i++) sum += a[a_off + i] * b[b_off + i];
    return sum;
#undef KOBLAS_DOT_DECLARE
#undef KOBLAS_DOT_STEP
}

KOBLAS_KERNEL double koblas_dense_ssqd(
    const double *a, int32_t a_off, const double *b, int32_t b_off, int32_t len
) {
#define KOBLAS_SSQD_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SSQD_STEP(q) \
    { const double d = a[a_off + i + q] - b[b_off + i + q]; s##q += d * d; }
    KOBLAS_REPEAT(KOBLAS_SSQD_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i + KOBLAS_ACCUMULATORS <= len; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SSQD_STEP) }
    }
    double sum = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
#undef KOBLAS_SSQD_DECLARE
#undef KOBLAS_SSQD_STEP
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

KOBLAS_KERNEL double koblas_dense_sum(const double *v, int32_t v_off, int32_t len) {
#define KOBLAS_SUM_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SUM_STEP(q) s##q += v[v_off + i + q];
    KOBLAS_REPEAT(KOBLAS_SUM_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i + KOBLAS_ACCUMULATORS <= len; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_SUM_STEP) }
    }
    double sum = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
    for (; i < len; i++) sum += v[v_off + i];
    return sum;
#undef KOBLAS_SUM_DECLARE
#undef KOBLAS_SUM_STEP
}

KOBLAS_KERNEL double koblas_dense_asum(const double *v, int32_t v_off, int32_t len) {
#define KOBLAS_ASUM_DECLARE(q) double s##q = 0.0;
#define KOBLAS_ASUM_STEP(q) s##q += fabs(v[v_off + i + q]);
    KOBLAS_REPEAT(KOBLAS_ASUM_DECLARE)
    int32_t i = 0;
    if (len >= KOBLAS_UNROLL_MIN) {
        for (; i + KOBLAS_ACCUMULATORS <= len; i += KOBLAS_ACCUMULATORS) { KOBLAS_REPEAT(KOBLAS_ASUM_STEP) }
    }
    double sum = KOBLAS_GATHER(s0, s1, s2, s3, s4, s5, s6, s7);
    for (; i < len; i++) sum += fabs(v[v_off + i]);
    return sum;
#undef KOBLAS_ASUM_DECLARE
#undef KOBLAS_ASUM_STEP
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

KOBLAS_KERNEL_BASELINE void koblas_dense_dot4(
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
#define KOBLAS_SPARSE_DOT_DECLARE(q) double s##q = 0.0;
#define KOBLAS_SPARSE_DOT_STEP(q) s##q += values[k + q] * dense[indices[k + q]];
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
