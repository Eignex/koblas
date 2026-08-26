package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.hostDispatchThresholds
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas

/**
 * Shared routing for a host sparse BLAS binding, the counterpart of the sparse factorization adapter.
 *
 * Below the gate a request is answered portably, since crossing into a native library costs more than the
 * work saves on a small problem. The gate counts stored entries rather than a dimension, because that is what
 * sparse work scales with.
 *
 * Only the products route natively. The libraries koblas binds carry a sparse-times-dense multiply and no
 * triangular solve over a caller's matrix, so [trsv] and [trsm] stay portable here rather than declaring a
 * hook nothing would fill.
 *
 * @param level2Min stored entries from which this binding multiplies natively, or null for the platform
 *   default. It moves the sparse product gate, not the dense level-2 one, which a vector-kernel platform sets
 *   beyond reach for a reason that does not carry here.
 */
public abstract class F64SparseBlasAdapter protected constructor(level2Min: Int? = null) : F64SparseBlas {
    /** Whether the binding resolved every symbol needed to multiply. */
    protected abstract val nativeAvailable: Boolean

    private val gate = hostDispatchThresholds(sparseProduct = level2Min).sparseProduct

    /** The portable routines, for everything the library will not take. */
    protected val portable: F64SparseBlas get() = F64ReferenceSparseLinearAlgebra

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    /**
     * Portable, deliberately. The libraries here multiply a sparse matrix by a dense one of any column count,
     * and what they charge for the call is fixed, so a single right-hand side has nothing to amortise it
     * over: measured against CHOLMOD this loses from the smallest matrices to the largest. [gemm] is where
     * that call pays.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature
    final override fun gemv(
        alpha: Double,
        a: F64SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ): Unit = portable.gemv(alpha, a, x, beta, y, transpose)

    @Suppress("LongParameterList") // the BLAS dgemm signature, plus the side the sparse operand sits on
    final override fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        right: Boolean,
    ) {
        // The libraries take the sparse operand on the left and a dense operand it can read as it stands, so
        // a product from the right or against a transposed dense operand is answered portably.
        if (a.nnz < gate || !nativeAvailable || right || transposeB) {
            return portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, right)
        }
        gemmNative(alpha, a, transposeA, b, beta, c)
    }

    /** `C = alpha · op(A) · B + beta · C` through the native library, with `B` read as it stands. */
    @Suppress("LongParameterList") // the BLAS dgemm signature less the flags this one does not take
    protected abstract fun gemmNative(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
    )

    final override fun trsv(
        a: F64SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ): Unit = portable.trsv(a, x, lower, transpose, unitDiag)

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    final override fun trsm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ): Unit = portable.trsm(a, b, lower, transpose, unitDiag, right, alpha)

    /**
     * Portable, deliberately. CHOLMOD has `cholmod_ssmult` and it was bound and measured; against the
     * portable product it is level at the smallest sizes and falls further behind as the operands grow,
     * because the result has to be built as a native matrix and copied back where the portable one writes
     * its arrays once. A gate says "native from here up" and there is no such point when the gap widens.
     */
    final override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix = portable.gemm(a, b)

    final override fun transpose(a: F64SparseMatrix): F64SparseMatrix = portable.transpose(a)
}
