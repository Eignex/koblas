package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64LuDecomposition
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseLinearAlgebra
import com.eignex.koblas.sparse.basis.F64BasisSolvers

/**
 * Every backend koblas will use for a piece of work, in one object you can hold. Immutable, and itself a
 * [F64LinearAlgebra] and a [F64SparseLinearAlgebra] by delegation.
 *
 * @property kernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property decompositions dense factorizations.
 * @property sparseKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property sparseDecompositions sparse factorizations.
 * @property basisSolvers simplex basis solvers, a half of their own beside [sparseDecompositions].
 */
public class F64Context(
    override val kernels: F64Kernels,
    public val blas: F64Blas,
    public val decompositions: F64Decompositions,
    override val sparseKernels: F64SparseKernels,
    public val sparseBlas: F64SparseBlas,
    public val sparseDecompositions: F64SparseDecompositions,
    public val basisSolvers: F64BasisSolvers,
) : F64LinearAlgebra,
    F64Blas by blas,
    F64Decompositions by decompositions,
    F64SparseLinearAlgebra,
    F64SparseBlas by sparseBlas,
    F64SparseDecompositions by sparseDecompositions,
    F64BasisSolvers by basisSolvers {

    /** The operation-level dispatch requirement for routes this context can inspect. */
    public var dispatchPolicy: F64DispatchPolicy = F64DispatchPolicy.AUTO
        private set

    /** The action this context takes for non-native inspected routes in automatic mode. */
    public var fallbackPolicy: F64FallbackPolicy = F64FallbackPolicy.ALLOW
        private set

    internal var fallbackWarning: (BackendRoute) -> Unit = {}
        private set

    @Suppress("LongParameterList") // the seven backend roles plus their execution policy
    internal constructor(
        kernels: F64Kernels,
        blas: F64Blas,
        decompositions: F64Decompositions,
        sparseKernels: F64SparseKernels,
        sparseBlas: F64SparseBlas,
        sparseDecompositions: F64SparseDecompositions,
        basisSolvers: F64BasisSolvers,
        dispatchPolicy: F64DispatchPolicy,
        fallbackPolicy: F64FallbackPolicy,
        fallbackWarning: (BackendRoute) -> Unit,
    ) : this(kernels, blas, decompositions, sparseKernels, sparseBlas, sparseDecompositions, basisSolvers) {
        this.dispatchPolicy = dispatchPolicy
        this.fallbackPolicy = fallbackPolicy
        this.fallbackWarning = fallbackWarning
    }

    /**
     * The distinct names of the backends that do the matrix work, joined, such as `"openblas+reference"`.
     * The vector-kernel halves are left out; [koblasInfo] prints both parts.
     */
    override val name: String
        get() = BackendSlot.matrixHalves.map { backendFor(it).name }.distinct().joinToString("+")

    /** True when every half is koblas's own, so the context calls out to nothing. */
    override val isPortable: Boolean get() = BackendSlot.entries.all { backendFor(it).isPortable }

    /** True when every half can run, which a context assembled from resolved backends always can. */
    override val isAvailable: Boolean get() = BackendSlot.entries.all { backendFor(it).isAvailable }

    /** The strongest half's priority, so a context is at least as preferred as the best thing in it. */
    override val priority: Int get() = BackendSlot.entries.maxOf { backendFor(it).priority }

    /**
     * A copy with the named halves replaced and the rest kept. A replaced [kernels] reaches the
     * inherited routines of halves that follow the installed context, which requires [installBackends];
     * a half built around kernels of its own always keeps them.
     */
    public fun with(
        kernels: F64Kernels = this.kernels,
        blas: F64Blas = this.blas,
        decompositions: F64Decompositions = this.decompositions,
        sparseKernels: F64SparseKernels = this.sparseKernels,
        sparseBlas: F64SparseBlas = this.sparseBlas,
        sparseDecompositions: F64SparseDecompositions = this.sparseDecompositions,
        basisSolvers: F64BasisSolvers = this.basisSolvers,
    ): F64Context = F64Context(
        kernels = kernels,
        blas = blas,
        decompositions = decompositions,
        sparseKernels = sparseKernels,
        sparseBlas = sparseBlas,
        sparseDecompositions = sparseDecompositions,
        basisSolvers = basisSolvers,
        dispatchPolicy = dispatchPolicy,
        fallbackPolicy = fallbackPolicy,
        fallbackWarning = fallbackWarning,
    )

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
    ) {
        if (enforcesRoutingPolicy) {
            val xLen = if (transpose) a.rows else a.cols
            val yLen = if (transpose) a.cols else a.rows
            requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
            requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
            beforeDispatch(F64RouteQuery.DenseGemv(a.rows, a.cols))
        }
        blas.gemv(alpha, a, x, beta, y, transpose)
    }

    override fun gemv(a: F64DenseMatrix, x: DoubleArray, transpose: Boolean): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: F64DenseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
    ) {
        if (enforcesRoutingPolicy) {
            val m = if (transposeA) a.cols else a.rows
            val k = if (transposeA) a.rows else a.cols
            val kB = if (transposeB) b.cols else b.rows
            val n = if (transposeB) b.rows else b.cols
            requireShape(k == kB) { "gemm: op(A) is ${m}x$k but op(B) is ${kB}x$n" }
            requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
            beforeDispatch(F64RouteQuery.DenseGemm(m, n, k))
        }
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
    }

    override fun gemm(a: F64DenseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
            beforeDispatch(F64RouteQuery.DenseLu(a.rows))
        }
        return decompositions.factor(a)
    }

    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "factor: matrix must be square, got ${a.rows}x${a.cols}" }
            requireShape(out.n == a.rows) { "factorInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
            beforeDispatch(F64RouteQuery.DenseLu(a.rows))
        }
        return decompositions.factorInto(a, out)
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: F64SparseMatrix,
        transposeA: Boolean,
        b: F64DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: F64DenseMatrix,
        right: Boolean,
    ) {
        if (enforcesRoutingPolicy) {
            val aRows = if (transposeA) a.cols else a.rows
            val aCols = if (transposeA) a.rows else a.cols
            val bRows = if (transposeB) b.cols else b.rows
            val bCols = if (transposeB) b.rows else b.cols
            val m = if (right) bRows else aRows
            val n = if (right) aCols else bCols
            requireShape(if (right) bCols == aRows else aCols == bRows) {
                val first = if (right) "${bRows}x$bCols" else "${aRows}x$aCols"
                val second = if (right) "${aRows}x$aCols" else "${bRows}x$bCols"
                "gemm: $first does not meet $second"
            }
            requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
            beforeDispatch(F64RouteQuery.SparseDenseGemm(a.nnz, right, transposeB))
        }
        sparseBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c, right)
    }

    override fun gemm(a: F64SparseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c, false)
        return c
    }

    override fun factor(a: F64SparseMatrix): F64SparseFactorization {
        if (enforcesRoutingPolicy) {
            requireSquare(a, "factor")
            beforeDispatch(F64RouteQuery.SparseLu(a.nnz))
        }
        return sparseDecompositions.factor(a)
    }

    override fun toString(): String = "F64Context($name)"
}
