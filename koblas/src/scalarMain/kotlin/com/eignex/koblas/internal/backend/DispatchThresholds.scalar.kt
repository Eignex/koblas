package com.eignex.koblas.internal.backend

/**
 * Defaults away from the JVM, where the portable kernels are scalar loops rather than SIMD, so levels 2,
 * 3 and LAPACK dispatch from the smallest size.
 */
internal actual val platformDispatchThresholds: DispatchThresholds =
    DispatchThresholds(
        level1 = NATIVE_LEVEL1_MIN,
        level2 = 0,
        level3 = 0,
        factorize = 0,
        symmetricFactorize = NATIVE_SYMMETRIC_FACTORIZE_MIN,
        sparseProduct = NATIVE_SPARSE_PRODUCT_MIN,
    )

/**
 * The stored-entry count from which a host library takes the symmetric sparse factorizations.
 *
 * The one factorization gate that is not zero here. The others are, because a foreign call costs little from
 * a native target and the portable kernels are scalar, so a library wins from the smallest sizes. These two
 * do not cross there: what the portable `L·Lᵀ` and `L·D·Lᵀ` lose to CHOLMOD is fill rather than call
 * overhead, and they take the ordering they are given. Measured on linuxX64 they win by three times at 214
 * stored entries and by a tenth at 387, and lose by a quarter at 484, so the crossing is near 417. See the
 * JVM threshold, which carries the rest of the reasoning and the same caveat about structure.
 */
private const val NATIVE_SYMMETRIC_FACTORIZE_MIN = 448

/**
 * The stored-entry count from which a host library takes the sparse matrix products.
 *
 * Its own gate rather than the `level2` one, which is zero here for the dense routines and would say the
 * library takes every product however small. That is what the measurement contradicts: against CHOLMOD's
 * `cholmod_sdmult` over eight right-hand sides the portable product wins by a tenth at 17 stored entries,
 * then loses by 1.35 times at 50, 1.8 at 109, 2.0 at 296, 2.2 at 919 and 1.7 at 11526, so the crossing sits
 * between 17 and 50. From the `sparseProductGate` suite on linuxX64 over two runs, with `trsv` trailing as a
 * control that stayed flat at 50 and favoured the library at 17, where the library still lost.
 *
 * Earlier than the JVM's gate, whose same 50-entry matrix is where a foreign call there is only just paid
 * for; a call costs less from a native target, so the library is worth reaching sooner.
 *
 * Only the matrix-matrix product reads this. A single right-hand side has nothing to amortise the call over
 * and lost at every size measured, and so did the sparse-times-sparse product, so both stay portable
 * whatever this says.
 */
private const val NATIVE_SPARSE_PRODUCT_MIN = 32

/** The run length from which a foreign call is worth making for a level-1 primitive. */
private const val NATIVE_LEVEL1_MIN = 32
