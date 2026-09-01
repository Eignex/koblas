package com.eignex.koblas

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64StridedMatrixView
import com.eignex.koblas.core.F64StridedVectorView
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64Kernels
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64LuDecomposition
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.sparse.F64BasisFactorizations
import com.eignex.koblas.sparse.F64GeneralSparseLu
import com.eignex.koblas.sparse.F64QuasiDefiniteLdl
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.F64SparseBlas
import com.eignex.koblas.sparse.F64SparseCholesky
import com.eignex.koblas.sparse.F64SparseDecompositionRoles
import com.eignex.koblas.sparse.F64SparseDecompositions
import com.eignex.koblas.sparse.F64SparseKernels
import com.eignex.koblas.sparse.F64SparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.F64SparseQr
import com.eignex.koblas.sparse.F64SparseQrFactorization
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
 * @param sparseDecompositions providers used to seed the derived compatibility composition.
 * @property basisSolvers simplex basis solvers, a half of their own beside [sparseDecompositions].
 *
 * [sparseDecompositions] is a derived compatibility composition of the selected general LU, Cholesky,
 * quasi-definite LDL, and QR roles.
 */
public class F64Context(
    override val kernels: F64Kernels,
    public val blas: F64Blas,
    public val decompositions: F64Decompositions,
    override val sparseKernels: F64SparseKernels,
    public val sparseBlas: F64SparseBlas,
    sparseDecompositions: F64SparseDecompositions,
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

    private var selectedGeneralSparseLu: F64GeneralSparseLu = sparseDecompositions.generalLuCapability()

    /** Provider selected for ordinary sparse LU. */
    public val generalSparseLu: F64GeneralSparseLu get() = selectedGeneralSparseLu

    private var selectedRepeatedSparseLu: F64RepeatedSparseLu? = sparseDecompositions as? F64RepeatedSparseLu

    /** Provider selected for repeated-pattern LU, or null when none was selected. */
    public val repeatedSparseLu: F64RepeatedSparseLu? get() = selectedRepeatedSparseLu

    private var selectedSparseCholesky: F64SparseCholesky = sparseDecompositions.choleskyCapability()

    /** Provider selected for sparse Cholesky. */
    public val sparseCholesky: F64SparseCholesky get() = selectedSparseCholesky

    private var selectedQuasiDefiniteLdl: F64QuasiDefiniteLdl = sparseDecompositions.quasiDefiniteLdlCapability()

    /** Provider selected for sparse quasi-definite, numerically unpivoted `L * D * L^T`. */
    public val quasiDefiniteLdl: F64QuasiDefiniteLdl get() = selectedQuasiDefiniteLdl

    private var selectedSparseQr: F64SparseQr = sparseDecompositions.qrCapability()

    /** Provider selected for sparse QR. */
    public val sparseQr: F64SparseQr get() = selectedSparseQr

    /** A compatibility operation surface derived from the four selected sparse factorization providers. */
    public val sparseDecompositions: F64SparseDecompositions by lazy {
        F64SparseDecompositionRoles(generalSparseLu, sparseCholesky, quasiDefiniteLdl, sparseQr)
    }

    private var selectedBasisFactorizations: F64BasisFactorizations =
        (sparseDecompositions as? F64BasisFactorizations) ?: com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra

    /** Provider selected for basis factorizations with column replacement. */
    public val basisFactorizations: F64BasisFactorizations get() = selectedBasisFactorizations

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
        generalSparseLu: F64GeneralSparseLu = sparseDecompositions.generalLuCapability(),
        repeatedSparseLu: F64RepeatedSparseLu? = sparseDecompositions as? F64RepeatedSparseLu,
        sparseCholesky: F64SparseCholesky = sparseDecompositions.choleskyCapability(),
        quasiDefiniteLdl: F64QuasiDefiniteLdl = sparseDecompositions.quasiDefiniteLdlCapability(),
        sparseQr: F64SparseQr = sparseDecompositions.qrCapability(),
        basisFactorizations: F64BasisFactorizations =
            (sparseDecompositions as? F64BasisFactorizations)
                ?: com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra,
    ) : this(kernels, blas, decompositions, sparseKernels, sparseBlas, sparseDecompositions, basisSolvers) {
        this.dispatchPolicy = dispatchPolicy
        this.fallbackPolicy = fallbackPolicy
        this.fallbackWarning = fallbackWarning
        selectedGeneralSparseLu = generalSparseLu
        selectedRepeatedSparseLu = repeatedSparseLu
        selectedSparseCholesky = sparseCholesky
        selectedQuasiDefiniteLdl = quasiDefiniteLdl
        selectedSparseQr = sparseQr
        selectedBasisFactorizations = basisFactorizations
    }

    /**
     * The distinct names of the backends that do the matrix work, joined, such as `"openblas+reference"`.
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
        generalSparseLu = if (sparseDecompositions === this.sparseDecompositions) {
            generalSparseLu
        } else {
            sparseDecompositions.generalLuCapability()
        },
        repeatedSparseLu = if (sparseDecompositions === this.sparseDecompositions) {
            repeatedSparseLu
        } else {
            sparseDecompositions as? F64RepeatedSparseLu
        },
        sparseCholesky = if (sparseDecompositions === this.sparseDecompositions) {
            sparseCholesky
        } else {
            sparseDecompositions.choleskyCapability()
        },
        quasiDefiniteLdl = if (sparseDecompositions === this.sparseDecompositions) {
            quasiDefiniteLdl
        } else {
            sparseDecompositions.quasiDefiniteLdlCapability()
        },
        sparseQr = if (sparseDecompositions === this.sparseDecompositions) {
            sparseQr
        } else {
            sparseDecompositions.qrCapability()
        },
        basisFactorizations = if (sparseDecompositions === this.sparseDecompositions) {
            basisFactorizations
        } else {
            (sparseDecompositions as? F64BasisFactorizations)
                ?: com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
        },
    )

    override fun gemv(
        alpha: Double,
        a: F64DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            val xLen = if (transpose) a.rows else a.cols
            val yLen = if (transpose) a.cols else a.rows
            requireShape(x.size == xLen) { "gemv: x length ${x.size} != $xLen" }
            requireShape(y.size == yLen) { "gemv: y length ${y.size} != $yLen" }
            beforeDispatch(F64RouteQuery.DenseGemv(a.rows, a.cols))
        }
        blas.gemv(alpha, a, x, beta, y, transpose, workspace)
    }

    @Suppress("LongParameterList") // the BLAS dgemv signature
    override fun gemv(
        alpha: Double,
        a: F64StridedMatrixView,
        x: F64StridedVectorView,
        beta: Double,
        y: F64StridedVectorView,
        transpose: Boolean,
    ) {
        if (enforcesRoutingPolicy) {
            val xLength = if (transpose) a.rows else a.cols
            val yLength = if (transpose) a.cols else a.rows
            requireShape(x.size == xLength) { "gemv: x length ${x.size} != $xLength" }
            requireShape(y.size == yLength) { "gemv: y length ${y.size} != $yLength" }
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
        workspace: Workspace?,
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
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)
    }

    override fun gemm(a: F64DenseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    @Suppress("LongParameterList") // the BLAS dgemm signature
    override fun gemm(
        alpha: Double,
        a: F64StridedMatrixView,
        transposeA: Boolean,
        b: F64StridedMatrixView,
        transposeB: Boolean,
        beta: Double,
        c: F64StridedMatrixView,
    ) {
        if (enforcesRoutingPolicy) {
            val m = if (transposeA) a.cols else a.rows
            val k = if (transposeA) a.rows else a.cols
            val otherK = if (transposeB) b.cols else b.rows
            val n = if (transposeB) b.rows else b.cols
            requireShape(k == otherK) { "gemm: op(A) is ${m}x$k but op(B) is ${otherK}x$n" }
            requireShape(c.rows == m && c.cols == n) { "gemm: C is ${c.rows}x${c.cols}, expected ${m}x$n" }
            beforeDispatch(F64RouteQuery.DenseGemm(m, n, k))
        }
        blas.gemm(alpha, a, transposeA, b, transposeB, beta, c)
    }

    override fun factor(a: F64DenseMatrix): F64LuDecomposition {
        if (enforcesRoutingPolicy) {
            beforeDispatch(F64RouteQuery.DenseLu(minOf(a.rows, a.cols)))
        }
        return decompositions.factor(a)
    }

    override fun factorInto(a: F64DenseMatrix, out: F64LuDecomposition): F64LuDecomposition {
        if (enforcesRoutingPolicy) {
            requireShape(out.rows == a.rows && out.cols == a.cols) {
                "factorInto: out is ${out.rows}x${out.cols}, expected ${a.rows}x${a.cols}"
            }
            beforeDispatch(F64RouteQuery.DenseLu(minOf(a.rows, a.cols)))
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
        workspace: Workspace?,
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
        sparseBlas.gemm(alpha, a, transposeA, b, transposeB, beta, c, right, workspace)
    }

    override fun gemm(a: F64SparseMatrix, b: F64DenseMatrix): F64DenseMatrix {
        val c = F64DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c, false)
        return c
    }

    override fun trsv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "trsv: matrix must be square, got ${a.rows}x${a.cols}" }
            requireShape(x.size == a.rows) { "trsv: x length ${x.size} != ${a.rows}" }
            beforeDispatch(
                F64RouteQuery.SparseTriangular(
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

    override fun trmv(a: F64SparseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "trmv: matrix must be square, got ${a.rows}x${a.cols}" }
            requireShape(x.size == a.rows) { "trmv: x length ${x.size} != ${a.rows}" }
            beforeDispatch(
                F64RouteQuery.SparseTriangular(
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
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "trsm: matrix must be square, got ${a.rows}x${a.cols}" }
            if (right) {
                requireShape(b.cols == a.rows) { "trsm right: B has ${b.cols} cols, expected ${a.rows}" }
            } else {
                requireShape(b.rows == a.rows) { "trsm: B has ${b.rows} rows, expected ${a.rows}" }
            }
            val rightHandSides = if (right) b.rows else b.cols
            beforeDispatch(
                F64RouteQuery.SparseTriangular(
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
        a: F64SparseMatrix,
        b: F64DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) {
        if (enforcesRoutingPolicy) {
            requireShape(a.rows == a.cols) { "trmm: matrix must be square, got ${a.rows}x${a.cols}" }
            if (right) {
                requireShape(b.cols == a.rows) { "trmm right: B has ${b.cols} cols, expected ${a.rows}" }
            } else {
                requireShape(b.rows == a.rows) { "trmm: B has ${b.rows} rows, expected ${a.rows}" }
            }
            beforeDispatch(
                F64RouteQuery.SparseTriangular(
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

    override fun factor(a: F64SparseMatrix): F64SparseLuFactorization {
        if (enforcesRoutingPolicy) {
            requireSquare(a, "factor")
            beforeDispatch(F64RouteQuery.SparseLu(a.nnz))
        }
        return sparseDecompositions.factor(a)
    }

    override fun qr(a: F64SparseMatrix): F64SparseQrFactorization {
        if (enforcesRoutingPolicy) {
            beforeDispatch(F64RouteQuery.SparseQr(a.nnz))
        }
        return sparseDecompositions.qr(a)
    }

    override fun toString(): String = "F64Context($name)"
}

private fun F64SparseDecompositions.generalLuCapability(): F64GeneralSparseLu =
    (this as? F64SparseDecompositionRoles)?.generalLu
        ?: (this as? F64GeneralSparseLu)
        ?: error("$name fills no general sparse LU role")

private fun F64SparseDecompositions.choleskyCapability(): F64SparseCholesky =
    (this as? F64SparseDecompositionRoles)?.choleskyProvider
        ?: (this as? F64SparseCholesky)
        ?: error("$name fills no sparse Cholesky role")

private fun F64SparseDecompositions.quasiDefiniteLdlCapability(): F64QuasiDefiniteLdl =
    (this as? F64SparseDecompositionRoles)?.quasiDefiniteLdlProvider
        ?: (this as? F64QuasiDefiniteLdl)
        ?: error("$name fills no sparse quasi-definite LDL role")

private fun F64SparseDecompositions.qrCapability(): F64SparseQr = (this as? F64SparseDecompositionRoles)?.qrProvider
    ?: (this as? F64SparseQr)
    ?: error("$name fills no sparse QR role")
