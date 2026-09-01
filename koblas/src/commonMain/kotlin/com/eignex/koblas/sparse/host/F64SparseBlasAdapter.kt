package com.eignex.koblas.sparse.host

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseBlas

/**
 * Shared routing for a host sparse BLAS binding, the counterpart of the sparse factorization adapter.
 * Only the products route natively by default. The libraries koblas currently binds carry no triangular
 * solve over a caller's matrix, so [trsv] and [trsm] stay portable here. They remain open for a provider that
 * can preserve the public storage, aliasing, singularity, and in-place contracts. Such a provider must also
 * override [route] for [F64RouteQuery.SparseTriangularSolve] through [triangularRoute].
 */
public abstract class F64SparseBlasAdapter protected constructor() :
    F64SparseBlas,
    F64RoutingBackend {
    /** Whether the binding resolved every symbol needed to multiply. */
    protected abstract val nativeAvailable: Boolean

    /** The portable routines, for everything the library will not take. */
    protected val portable: F64SparseBlas get() = F64ReferenceSparseLinearAlgebra

    override val isAvailable: Boolean get() = nativeAvailable

    override val isPortable: Boolean get() = false

    override fun route(query: F64RouteQuery): BackendRoute? {
        if (query is F64RouteQuery.SparseTriangularSolve) {
            return portableRoute(query, this, portable.name, BackendRouteReason.UNSUPPORTED_OPERATION)
        }
        if (query !is F64RouteQuery.SparseDenseGemm) return null
        val native = nativeRoute(query, this, portable.name)
        if (native.execution != BackendExecution.NATIVE || (!query.right && !query.transposeDense)) {
            return native
        }
        return native.copy(
            execution = BackendExecution.PORTABLE,
            executor = portable.name,
            reason = BackendRouteReason.UNSUPPORTED_ARGUMENTS,
        )
    }

    /**
     * Builds the route for a specialized sparse triangular solve.
     *
     * [supported] describes argument forms the provider cannot execute, independently of the crossover.
     * Subclasses overriding [trsv] or [trsm] should return this from [route] for the matching queries.
     */
    protected fun triangularRoute(
        query: F64RouteQuery.SparseTriangularSolve,
        supported: Boolean = true,
    ): BackendRoute {
        val native = nativeRoute(query, this, portable.name)
        if (supported || native.execution != BackendExecution.NATIVE) return native
        return native.copy(
            execution = BackendExecution.PORTABLE,
            executor = portable.name,
            reason = BackendRouteReason.UNSUPPORTED_ARGUMENTS,
        )
    }

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
        workspace: Workspace?,
    ) {
        // The libraries take the sparse operand on the left and a dense operand it can read as it stands, so
        // a product from the right or against a transposed dense operand is answered portably.
        if (!nativeAvailable || right || transposeB) {
            return portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, right, workspace)
        }
        gemmNative(alpha, a, transposeA, b, beta, c, workspace)
    }

    /** `C = alpha · op(A) · B + beta · C` through the native library, with `B` read as it stands. [workspace]
     *  reuses portable staging on the software fallback the native call keeps for itself. */
    @Suppress("LongParameterList") // the BLAS dgemm signature less the flags this one does not take
    protected abstract fun gemmNative(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        beta: Double,
        c: F64DenseMatrix,
        workspace: Workspace?,
    )

    override fun trsv(
        a: F64SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ): Unit = portable.trsv(a, x, lower, transpose, unitDiag)

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ): Unit = portable.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)

    /**
     * Portable, deliberately. CHOLMOD has `cholmod_ssmult` and it was bound and measured; against the
     * portable product it is level at the smallest sizes and falls further behind as the operands grow,
     * because the result has to be built as a native matrix and copied back where the portable one writes
     * its arrays once.
     */
    final override fun gemm(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix = portable.gemm(a, b)

    final override fun transpose(a: F64SparseMatrix): F64SparseMatrix = portable.transpose(a)
}
