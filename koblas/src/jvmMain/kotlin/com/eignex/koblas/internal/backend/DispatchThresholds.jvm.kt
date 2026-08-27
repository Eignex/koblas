package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.simdAvailable

/**
 * JVM defaults, chosen from [simdAvailable] because they hold only while the Vector API kernels are there.
 * The incubator module is opt-in, and without it the crossovers move down to the scalar ones.
 *
 * Whether the module resolved is fixed for the process.
 */
internal actual val platformDispatchThresholds: DispatchThresholds
    get() = if (simdAvailable) SIMD_THRESHOLDS else SCALAR_THRESHOLDS

/** For a JVM running the Vector API kernels. */
private val SIMD_THRESHOLDS = DispatchThresholds(
    level1 = Int.MAX_VALUE,
    level2 = Int.MAX_VALUE,
    level3 = JVM_LEVEL3_MIN,
    factorize = JVM_FACTORIZE_MIN,
    symmetricFactorize = JVM_SYMMETRIC_FACTORIZE_MIN,
    sparseProduct = JVM_SPARSE_PRODUCT_MIN,
    sparseQr = JVM_SPARSE_QR_MIN,
)

/**
 * For a JVM running the scalar loops, where the host library wins from the smallest sizes.
 *
 * The symmetric factorizations are the exception, and keep the gate the vector path uses. What they lose to
 * CHOLMOD is fill rather than the arithmetic in a kernel, so having scalar loops instead of vector ones does
 * not move where they cross.
 */
private val SCALAR_THRESHOLDS = DispatchThresholds(
    level1 = JVM_SCALAR_LEVEL1_MIN,
    level2 = 0,
    level3 = 0,
    factorize = 0,
    symmetricFactorize = JVM_SYMMETRIC_FACTORIZE_MIN,
    sparseProduct = JVM_SPARSE_PRODUCT_MIN,
    sparseQr = JVM_SCALAR_SPARSE_QR_MIN,
)

/**
 * The run length from which a foreign call is worth making for a level-1 primitive, later than the native
 * one because the call costs more from here.
 */
private const val JVM_SCALAR_LEVEL1_MIN = 128

/** The matrix size from which the host BLAS takes the level-3 routines. */
private const val JVM_LEVEL3_MIN = 16

/**
 * The matrix size from which the host LAPACK takes the factorizations. Nothing crosses at 64: the LU
 * factorizations cross here, and Cholesky and QR only at 256, so this dispatches those two a little early.
 */
private const val JVM_FACTORIZE_MIN = 128

/**
 * The stored-entry count from which a host library takes the symmetric sparse factorizations.
 *
 * Far later than [JVM_FACTORIZE_MIN], which is where the general factorizations cross. Against CHOLMOD the
 * portable `L·Lᵀ` and `L·D·Lᵀ` win by three times at 214 stored entries and by a quarter at 387, and lose by
 * a tenth at 484 and by nearly twice at 595, so the curves cross around 458. The native targets cross a
 * little earlier, near 417, their portable kernels being scalar; this sits inside the bracket both were
 * measured in. From the `symmetricGate` suite, with `transpose` trailing as a control that stayed flat.
 *
 * One number cannot be right for every matrix. Those came from a one-percent-density random matrix, which
 * fills hardest and so favours a library with a fill-reducing ordering most; the portable factorization takes
 * the ordering it is given, and on a banded or tree-structured matrix it keeps winning well past this.
 */
private const val JVM_SYMMETRIC_FACTORIZE_MIN = 448

/**
 * The stored-entry count from which a host library takes the sparse matrix products.
 *
 * Its own gate rather than the `level2` one, which the vector path sets beyond reach because the compiled-in
 * kernels win the dense level-2 routines; there are no vector kernels for a sparse product, so that
 * reasoning does not carry and reading it here would leave the binding unreachable.
 *
 * Against CHOLMOD's `cholmod_sdmult` the portable product over eight right-hand sides is level at 50 stored
 * entries and behind by half again at 109, 919 and 11526, so the crossing is just above 50. Measured through
 * the `sparseProductGate` suite, with `trsv` trailing as a control that stayed within a percent throughout.
 *
 * Only the matrix-matrix product reads this. A single right-hand side has nothing to amortise the call over
 * and lost at every size measured, and so did the sparse-times-sparse product, so both stay portable
 * whatever this says.
 */
private const val JVM_SPARSE_PRODUCT_MIN = 64

/**
 * Stored entries from which SPQR takes sparse QR. At 153 entries it was 1.69 times slower to factor and
 * 1.15 times slower to factor and solve; at 467 it was 2.93 and 2.32 times faster. The portable sparse QR
 * does not call the Vector API, so the same crossing applies when the JVM's dense kernels are scalar.
 * Measured through `sparseQrGate` with `R` escaping the factor-only row so the JIT cannot discard it.
 */
private const val JVM_SPARSE_QR_MIN = 384

/** The sparse QR is scalar on both JVM execution modes. */
private const val JVM_SCALAR_SPARSE_QR_MIN = JVM_SPARSE_QR_MIN
