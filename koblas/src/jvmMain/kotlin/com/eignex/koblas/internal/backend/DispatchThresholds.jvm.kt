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
)

/** For a JVM running the scalar loops, where the host library wins from the smallest sizes. */
private val SCALAR_THRESHOLDS = DispatchThresholds(
    level1 = JVM_SCALAR_LEVEL1_MIN,
    level2 = 0,
    level3 = 0,
    factorize = 0,
)

/**
 * One set whether or not the Vector API resolved.
 *
 * What a sparse routine loses to its library is fill and foreign-call overhead rather than the arithmetic in
 * a dense kernel, so having scalar loops instead of vector ones does not move where it crosses. Reading these
 * off the dense gates made the general sparse LU dispatch from zero entries on a JVM without the incubator
 * module and from 128 with it, on a crossing neither number was measured against.
 */
internal actual val platformSparseDispatchThresholds: SparseDispatchThresholds = SparseDispatchThresholds(
    factorize = JVM_SPARSE_FACTORIZE_MIN,
    symmetric = JVM_SYMMETRIC_FACTORIZE_MIN,
    qr = JVM_SPARSE_QR_MIN,
    product = JVM_SPARSE_PRODUCT_MIN,
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

/** The sparse general LU against UMFPACK, in stored entries. Inherited from the dense gate, not measured. */
private const val JVM_SPARSE_FACTORIZE_MIN = 128

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
 * Measured through `sparseQrGate`. SPQR copies `R` and the Householder form out of the library at
 * factorization time rather than holding them, so the figures include that copy and reading the factors
 * afterwards costs nothing.
 */
private const val JVM_SPARSE_QR_MIN = 384
