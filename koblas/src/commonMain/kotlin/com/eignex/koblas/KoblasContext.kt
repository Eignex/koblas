package com.eignex.koblas

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.StridedMatrixView
import com.eignex.koblas.StridedVectorView
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Kernels
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.GeneralSparseLu
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparseLinearAlgebra
import com.eignex.koblas.sparse.SparseLuFactorization
import com.eignex.koblas.sparse.basis.BasisSolvers

/**
 * Every backend koblas will use for a piece of work, in one object you can hold. Immutable, and itself a
 * [Blas] and a [SparseLinearAlgebra] by delegation.
 *
 * @property kernels dense vector-vector routines; every dense inner loop bottoms out here.
 * @property blas dense matrix routines.
 * @property sparseKernels sparse vector-vector routines.
 * @property sparseBlas sparse matrix routines.
 * @property basisSolvers simplex basis solvers supplied by HFactor.
 * @param roles the retained HFactor sparse LU provider.
 * @property dispatchPolicy the operation-level dispatch requirement for routes this context can inspect.
 * @property fallbackPolicy the action taken for non-native inspected routes in automatic mode.
 * @property fallbackWarning notified for each fallback under [FallbackPolicy.WARN].
 *
 * Sparse LU and basis solving are the remaining factorization roles.
 */
@Suppress("LongParameterList") // the backend halves, resolved roles, and execution policy
public class KoblasContext internal constructor(
    override val kernels: Kernels,
    public val blas: Blas,
    override val sparseKernels: SparseKernels,
    public val sparseBlas: SparseBlas,
    public val basisSolvers: BasisSolvers,
    private val roles: SparseRoles,
    public val dispatchPolicy: DispatchPolicy = DispatchPolicy.AUTO,
    public val fallbackPolicy: FallbackPolicy = FallbackPolicy.ALLOW,
    internal val fallbackWarning: (BackendRoute) -> Unit = {},
) : Blas by blas,
    SparseLinearAlgebra,
    SparseBlas by sparseBlas,
    GeneralSparseLu,
    BasisSolvers by basisSolvers {

    /**
     * Creates a context from explicit backend halves.
     */
    public constructor(
        kernels: Kernels,
        blas: Blas,
        sparseKernels: SparseKernels,
        sparseBlas: SparseBlas,
        generalSparseLu: GeneralSparseLu,
        basisSolvers: BasisSolvers,
    ) : this(
        kernels,
        blas,
        sparseKernels,
        sparseBlas,
        basisSolvers,
        SparseRoles(generalSparseLu),
    )

    /** Provider selected for ordinary sparse LU. */
    public val generalSparseLu: GeneralSparseLu get() = roles.generalLu

    /**
     * The distinct names of the backends that do the matrix work, joined, such as `"reference+hfactor"`.
     * The vector-kernel halves are left out; [koblasInfo] prints both parts.
     */
    override val name: String
        get() = BackendSlot.matrixHalves.map { it.from(this).name }.distinct().joinToString("+")

    /** True when every half is koblas's own, so the context calls out to nothing. */
    override val isPortable: Boolean get() = BackendSlot.contextHalves.all { it.from(this).isPortable }

    /** True when every half can run, which a context assembled from resolved backends always can. */
    override val isAvailable: Boolean get() = BackendSlot.contextHalves.all { it.from(this).isAvailable }

    /** The strongest half's priority, so a context is at least as preferred as the best thing in it. */
    override val priority: Int get() = BackendSlot.contextHalves.maxOf { it.from(this).priority }

    /** The first half that says why it cannot run, since a context is unusable as soon as one of them is. */
    override val unavailableReason: String?
        get() = BackendSlot.contextHalves.firstNotNullOfOrNull { it.from(this).unavailableReason }

    /**
     * A copy with the named halves replaced and the rest kept. A replaced [kernels] reaches the
     * inherited routines of halves that follow the installed context, which requires [installBackends];
     * a half built around kernels of its own always keeps them.
     */
    public fun with(
        kernels: Kernels = this.kernels,
        blas: Blas = this.blas,
        sparseKernels: SparseKernels = this.sparseKernels,
        sparseBlas: SparseBlas = this.sparseBlas,
        generalSparseLu: GeneralSparseLu = this.generalSparseLu,
        basisSolvers: BasisSolvers = this.basisSolvers,
    ): KoblasContext = KoblasContext(
        kernels = kernels,
        blas = blas,
        sparseKernels = sparseKernels,
        sparseBlas = sparseBlas,
        basisSolvers = basisSolvers,
        dispatchPolicy = dispatchPolicy,
        fallbackPolicy = fallbackPolicy,
        fallbackWarning = fallbackWarning,
        roles = SparseRoles(generalSparseLu),
    )

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            requireGemvShape(a, transpose, x.size, y.size)
            beforeDispatch(RouteQuery.DenseGemv(a.rows, a.cols))
        }
        blas.gemv(alpha, a, x, beta, y, transpose, workspace)
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: StridedMatrixView,
        x: StridedVectorView,
        beta: Double,
        y: StridedVectorView,
        transpose: Boolean,
    ) {
        if (enforcesRoutingPolicy) {
            requireGemvShape(a, transpose, x.size, y.size)
            beforeDispatch(RouteQuery.DenseGemv(a.rows, a.cols))
        }
        blas.gemv(alpha, a, x, beta, y, transpose)
    }

    override fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
            beforeDispatch(RouteQuery.DenseGemm(m, n, k))
        }
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)
    }

    override fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature
    override fun gemm(
        alpha: Double,
        a: StridedMatrixView,
        transposeA: Boolean,
        b: StridedMatrixView,
        transposeB: Boolean,
        beta: Double,
        c: StridedMatrixView,
    ) {
        if (enforcesRoutingPolicy) {
            val (m, k, n) = requireGemmShape(a, transposeA, b, transposeB, c)
            beforeDispatch(RouteQuery.DenseGemm(m, n, k))
        }
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
    }

    @Suppress("LongParameterList")
    override fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        right: Boolean,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            // Multiplying the dense operand by the sparse one from the right is this product with the
            // operands the other way round, so the same derivation answers both.
            if (right) {
                requireGemmShape(b, transposeB, a, transposeA, c)
            } else {
                requireGemmShape(a, transposeA, b, transposeB, c)
            }
            beforeDispatch(RouteQuery.SparseDenseGemm(a.nnz, right, transposeB))
        }
        sparseBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c, right, workspace)
    }

    override fun gemm(a: SparseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c, false)
        return c
    }

    override fun trsv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (enforcesRoutingPolicy) {
            requireSquare(a, "trsv")
            requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
            beforeDispatch(
                RouteQuery.SparseTriangular(
                    a.nnz,
                    kind = SparseTriangularKind.SOLVE,
                    lower = lower,
                    transpose = transpose,
                    unitDiagonal = unitDiag,
                ),
            )
        }
        sparseBlas.trsv(a, x, lower, transpose, unitDiag)
    }

    override fun trmv(a: SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (enforcesRoutingPolicy) {
            requireSquare(a, "trmv")
            requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
            beforeDispatch(
                RouteQuery.SparseTriangular(
                    a.nnz,
                    kind = SparseTriangularKind.MULTIPLY,
                    lower = lower,
                    transpose = transpose,
                    unitDiagonal = unitDiag,
                ),
            )
        }
        sparseBlas.trmv(a, x, lower, transpose, unitDiag)
    }

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            requireTriangularMatrixShape(a, b, right, "trsm")
            val rightHandSides = if (right) b.rows else b.cols
            beforeDispatch(
                RouteQuery.SparseTriangular(
                    a.nnz,
                    kind = SparseTriangularKind.SOLVE,
                    rightHandSides = rightHandSides,
                    lower = lower,
                    right = right,
                    transpose = transpose,
                    unitDiagonal = unitDiag,
                ),
            )
        }
        sparseBlas.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        a: SparseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        if (enforcesRoutingPolicy) {
            requireTriangularMatrixShape(a, b, right, "trmm")
            beforeDispatch(
                RouteQuery.SparseTriangular(
                    a.nnz,
                    kind = SparseTriangularKind.MULTIPLY,
                    rightHandSides = if (right) b.rows else b.cols,
                    lower = lower,
                    right = right,
                    transpose = transpose,
                    unitDiagonal = unitDiag,
                ),
            )
        }
        sparseBlas.trmm(a, b, lower, transpose, unitDiag, right, alpha)
    }

    override fun factor(a: SparseMatrix): SparseLuFactorization {
        if (enforcesRoutingPolicy) {
            requireSquare(a, "factor")
            beforeDispatch(RouteQuery.SparseLu(a.nnz))
        }
        return generalSparseLu.factor(a)
    }

    override fun toString(): String = "KoblasContext($name)"
}

/**
 * The six sparse factorization providers a context selects, held together because they are selected
 * together. Derived once, at whichever constructor the context came in through: the internal one is handed
 * roles its caller resolved from the registry or a builder, and the public one is handed a composition to
 * read them out of.
 *
 * This is the internal representation shared by registry and builder assembly.
 */
internal class SparseRoles(val generalLu: GeneralSparseLu)
