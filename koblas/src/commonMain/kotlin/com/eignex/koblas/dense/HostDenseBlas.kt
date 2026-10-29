@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
)

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.RegionBlas
import com.eignex.koblas.vendor.RouteKind

/**
 * A dense implementation that can say what one of its own calls executes.
 *
 * The seam an engine holds, so that a composition which sometimes reaches a host library answers the question
 * for the call that was actually made rather than letting the portable description stand for both.
 */
internal interface RoutedDenseBlas : DenseBlas {
    /** What a call of [operation] with the facts in [call] executes. */
    fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute
}

/**
 * The name a composed host route carries as its scheduling, with the library named beside it as a component.
 *
 * Deliberately not the vendor's name on its own. A call served this way is Koblas's validation and alias
 * staging around one whole vendor entry point, and the two are different amounts of this library's code; the
 * component is what says which library and which symbol ran.
 */
internal const val HOST_SCHEDULING: String = "host-dense"

/** The copy a permitted alias is staged into before a whole-call binding, which cannot take one. */
internal const val HOST_STAGING: String = "host-stage/alias"

/** The packing a borrowed region takes before a binding that has no in-place region entry points. */
internal const val HOST_REGION_STAGING: String = "host-stage/region"

/** Platform eligibility with primitive extents, avoiding boxed dimensions on the ordinary call path. */
internal fun interface DenseHostEligibility {
    fun eligible(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean

    /** Borrowed operands can need additional packing before a binding receives them. */
    fun eligibleBorrowed(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean =
        eligible(operation, rows, columns, depth)
}

/**
 * Optional whole-call native acceleration with shared validation, alias handling and portable fallbacks.
 * The supplied platform policy owns crossovers; exports and zero-work rules remain correctness guards.
 * Region bindings borrow offsets and leading dimensions; other bindings receive contiguous staged copies.
 */
internal class HostDenseBlas(
    private val portable: PortableDenseBlas,
    private val host: Blas,
    private val eligible: DenseHostEligibility,
) : RoutedDenseBlas {

    /** The binding's in-place region entry points, or null where a borrowed operand has to be packed for it. */
    private val regions: RegionBlas? = host as? RegionBlas

    /** Symbol availability is independent of the platform's performance policy. */
    private fun host(operation: DenseMatrixOperation, eligibleCall: Boolean): Boolean =
        eligibleCall && HostDenseCall.entryPointFor(operation) in host.directlyImplemented

    private fun eligibleCall(
        operation: DenseMatrixOperation,
        rows: Int,
        columns: Int,
        depth: Int = 1,
        alpha: Double = 1.0,
        borrowed: Boolean = false,
    ): Boolean = alpha != 0.0 && rows > 0 && columns > 0 && depth > 0 &&
        permitted(operation, rows, columns, depth, borrowed)

    private fun permitted(
        operation: DenseMatrixOperation,
        rows: Int,
        columns: Int,
        depth: Int,
        borrowed: Boolean,
    ): Boolean {
        val policy = eligible
        return if (borrowed) {
            policy.eligibleBorrowed(operation, rows, columns, depth)
        } else {
            policy.eligible(operation, rows, columns, depth)
        }
    }

    override fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        if (HostDenseCall.needsDepth(operation) && call.depth == null) return portable.routeOf(operation, call)
        val permitted = eligibleCall(
            operation,
            call.rows,
            call.columns,
            call.depth ?: 1,
            call.alpha,
            call.borrowed,
        )
        if (!host(operation, permitted)) return portable.routeOf(operation, call)
        val entry = requireNotNull(HostDenseCall.entryPointFor(operation))
        // Work, whole-entry-point eligibility and exports are settled above; only platform transfer remains.
        val hostRoute = host.routeOf(entry)
        val components = ArrayList<String>(2)
        if (call.borrowed && regions == null) components.add(HOST_REGION_STAGING)
        if (call.aliased) components.add(HOST_STAGING)
        hostRoute.adapter?.let { components.add("host-adapter/$it") }
        components.add("${host.vendor.vendorName.lowercase()}/${entry.entryPoint}")
        return DenseMatrixRoute(
            operation = operation,
            kind = RouteKind.Direct,
            scheduling = HOST_SCHEDULING,
            entryPoint = operation.entryPoint,
            components = components,
            executionGroup = 0,
            host = hostRoute,
        )
    }

    // A binding with region entry points takes borrowed storage as it lies, so only an input overlapping the
    // destination is snapshotted, as the portable path does; a binding without them gets packed operands.

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        val op = if (transpose) DenseMatrixOperation.GemvTransposed else DenseMatrixOperation.Gemv
        val eligibleCall = eligibleCall(op, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix)
        val direct = regions
        when {
            !host(op, eligibleCall) -> portable.gemv(alpha, a, x, beta, y, transpose, workspace)

            direct == null -> regionGemv(this, alpha, a, x, beta, y, transpose, workspace)

            else -> {
                requireGemvOperands(a, transpose, x.size, y.size)
                stagedRegion(workspace, a, y) { sa ->
                    staged(workspace, x, x === y) { sx ->
                        direct.gemv(alpha, sa, transpose, sx.asVector(), beta, y.asVector())
                    }
                }
            }
        }
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace?,
    ) {
        val op = DenseMatrixOperation.Gemm
        val eligibleCall = eligibleCall(
            op,
            c.rows,
            c.cols,
            if (transposeA) a.rows else a.cols,
            alpha,
            borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
        )
        val direct = regions
        when {
            !host(op, eligibleCall) -> portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)

            direct == null -> regionGemm(this, alpha, a, transposeA, b, transposeB, beta, c, workspace)

            else -> {
                requireGemmOperands(a, transposeA, b, transposeB, c)
                stagedRegion(workspace, a, c) { sa ->
                    stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                        direct.gemm(alpha, sa, transposeA, sb, transposeB, beta, c)
                    }
                }
            }
        }
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val op = DenseMatrixOperation.Syrk
        val eligibleCall = eligibleCall(
            op,
            c.rows,
            c.cols,
            if (transpose) a.rows else a.cols,
            alpha,
            borrowed = a !is DenseMatrix || c !is DenseMatrix,
        )
        val direct = regions
        when {
            !host(op, eligibleCall) -> portable.syrk(alpha, a, transpose, beta, c, lower, workspace)

            direct == null -> regionSyrk(this, alpha, a, transpose, beta, c, lower, workspace)

            else -> {
                requireSyrkOperands(a, transpose, c)
                stagedRegion(workspace, a, c) { sa ->
                    direct.syrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
                }
            }
        }
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        val op = if (transpose) DenseMatrixOperation.TrsvTransposed else DenseMatrixOperation.Trsv
        val eligibleCall = eligibleCall(op, a.rows, a.cols, borrowed = a !is DenseMatrix)
        val direct = regions
        when {
            !host(op, eligibleCall) -> portable.trsv(a, x, lower, transpose, unitDiag, workspace)

            direct == null -> regionTrsv(this, a, x, lower, transpose, unitDiag, workspace)

            else -> {
                requireTriangularVectorOperands(a, x.size, "trsv")
                stagedRegion(workspace, a, x) { sa ->
                    direct.trsv(sa, triangle(lower, unitDiag), transpose, x.asVector())
                }
            }
        }
    }

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) {
        val op = DenseMatrixOperation.Trsm
        val eligibleCall = eligibleCall(
            op,
            b.rows,
            b.cols,
            a.rows,
            alpha,
            borrowed = a !is DenseMatrix || b !is DenseMatrix,
        )
        val direct = regions
        when {
            !host(op, eligibleCall) -> portable.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)

            direct == null -> regionTrsm(this, a, b, lower, transpose, unitDiag, right, alpha, workspace)

            else -> {
                requireTriangularMatrixOperands(a, b, right, "trsm")
                stagedRegion(workspace, a, b) { sa ->
                    direct.trsm(alpha, sa, triangle(lower, unitDiag), transpose, b, rightSide = right)
                }
            }
        }
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        val operation = if (transpose) DenseMatrixOperation.GemvTransposed else DenseMatrixOperation.Gemv
        if (!host(operation, eligibleCall(operation, a.rows, a.cols, alpha = alpha))) {
            portable.gemv(alpha, a, x, beta, y, transpose, workspace)
            return
        }
        staged(workspace, a, a.values === y) { sa ->
            staged(workspace, x, x === y) { sx ->
                host.gemv(alpha, sa, transpose, sx.asVector(), beta, y.asVector())
            }
        }
    }

    override fun transpose(a: DenseMatrix): DenseMatrix = portable.transpose(a)

    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (!host(DenseMatrixOperation.Symv, eligibleCall(DenseMatrixOperation.Symv, a.rows, a.cols, alpha = alpha))) {
            portable.symv(alpha, a, x, beta, y, lower, workspace)
            return
        }
        staged(workspace, a, a.values === y) { sa ->
            staged(workspace, x, x === y) { sx ->
                host.symv(alpha, sa, symmetricStructure(lower), sx.asVector(), beta, y.asVector())
            }
        }
    }

    /**
     * The rank updates, whose destination is the matrix and whose inputs are the vectors.
     *
     * [ger] takes plain arrays and the symmetric pair takes vectors, which is [DenseBlas]'s shape rather than
     * anything this layer chooses. A strided vector needs no staging of its own: an increment is what BLAS
     * addresses a vector with, so the binding passes the caller's spacing through unchanged.
     */
    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, workspace: Workspace?) {
        requireGerOperands(x.size, y.size, a)
        if (!host(DenseMatrixOperation.Ger, eligibleCall(DenseMatrixOperation.Ger, a.rows, a.cols, alpha = alpha))) {
            portable.ger(alpha, x, y, a, workspace)
            return
        }
        staged(workspace, x, x === a.values) { sx ->
            staged(workspace, y, y === a.values) { sy ->
                host.ger(alpha, sx.asVector(), sy.asVector(), a)
            }
        }
    }

    // Whole-matrix bindings do not expose window offsets or leading dimensions.
    @Suppress("LongParameterList") // the rank update and its three independently addressed windows
    override fun ger(
        alpha: Double,
        x: DoubleArray,
        y: DoubleArray,
        a: DenseMatrix,
        rowOffset: Int,
        columnOffset: Int,
        rows: Int,
        columns: Int,
        xOffset: Int,
        yOffset: Int,
        workspace: Workspace?,
    ): Unit = portable.ger(alpha, x, y, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset, workspace)

    @Suppress("LongParameterList") // the BLAS dsyr signature plus the workspace
    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean, workspace: Workspace?) {
        requireSyrOperands(a, x.size, "syr")
        if (!host(DenseMatrixOperation.Syr, eligibleCall(DenseMatrixOperation.Syr, a.rows, a.cols, alpha = alpha))) {
            portable.syr(alpha, x, a, lower, workspace)
            return
        }
        staged(workspace, x, x.values === a.values) { sx ->
            host.syr(alpha, sx, a, symmetricStructure(lower))
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2 signature plus the workspace
    override fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2Operands(a, x.size, y.size, "syr2")
        if (!host(DenseMatrixOperation.Syr2, eligibleCall(DenseMatrixOperation.Syr2, a.rows, a.cols, alpha = alpha))) {
            portable.syr2(alpha, x, y, a, lower, workspace)
            return
        }
        staged(workspace, x, x.values === a.values) { sx ->
            staged(workspace, y, y.values === a.values) { sy ->
                host.syr2(alpha, sx, sy, a, symmetricStructure(lower))
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
    override fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        val operation = if (transpose) DenseMatrixOperation.TrsvTransposed else DenseMatrixOperation.Trsv
        triangularVector(a, x, lower, transpose, unitDiag, operation, "trsv", workspace)
    }

    @Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
    override fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        val operation = if (transpose) DenseMatrixOperation.TrmvTransposed else DenseMatrixOperation.Trmv
        triangularVector(a, x, lower, transpose, unitDiag, operation, "trmv", workspace)
    }

    @Suppress("LongParameterList") // the BLAS triangular signature, the operation it names and the workspace
    private fun triangularVector(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        operation: DenseMatrixOperation,
        what: String,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, what)
        val solve = operation == DenseMatrixOperation.Trsv || operation == DenseMatrixOperation.TrsvTransposed
        if (!host(operation, eligibleCall(operation, a.rows, a.cols))) {
            if (solve) {
                portable.trsv(a, x, lower, transpose, unitDiag, workspace)
            } else {
                portable.trmv(a, x, lower, transpose, unitDiag, workspace)
            }
            return
        }
        // The triangle is the only operand that can share the destination, since x is the destination.
        staged(workspace, a, a.values === x) { sa ->
            val structure = triangle(lower, unitDiag)
            if (solve) {
                host.trsv(sa, structure, transpose, x.asVector())
            } else {
                host.trmv(sa, structure, transpose, x.asVector())
            }
        }
    }

    // Contiguous column-major operands need staging only when they overlap the destination.

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
        requireGemmOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        val gemm = DenseMatrixOperation.Gemm
        if (!host(gemm, eligibleCall(gemm, c.rows, c.cols, depth, alpha))) {
            portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)
            return
        }
        stagedOperands(workspace, a, b, c) { sa, sb ->
            host.gemm(alpha, sa, transposeA, sb, transposeB, beta, c)
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        val gemmt = DenseMatrixOperation.Gemmt
        if (!host(gemmt, eligibleCall(gemmt, c.rows, c.cols, depth, alpha))) {
            portable.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, workspace)
            return
        }
        stagedOperands(workspace, a, b, c) { sa, sb ->
            host.gemmt(alpha, sa, transposeA, sb, transposeB, beta, c, symmetricStructure(lower))
        }
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyrkOperands(a, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        val syrk = DenseMatrixOperation.Syrk
        if (!host(syrk, eligibleCall(syrk, c.rows, c.cols, depth, alpha))) {
            portable.syrk(alpha, a, transpose, beta, c, lower, workspace)
            return
        }
        staged(workspace, a, a.values === c.values) { sa ->
            host.syrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2kOperands(a, b, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        val syr2k = DenseMatrixOperation.Syr2k
        if (!host(syr2k, eligibleCall(syr2k, c.rows, c.cols, depth, alpha))) {
            portable.syr2k(alpha, a, b, transpose, beta, c, lower, workspace)
            return
        }
        stagedOperands(workspace, a, b, c) { sa, sb ->
            host.syr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        val symm = DenseMatrixOperation.Symm
        if (!host(symm, eligibleCall(symm, c.rows, c.cols, a.rows, alpha))) {
            portable.symm(alpha, a, b, beta, c, lower, right, workspace)
            return
        }
        stagedOperands(workspace, a, b, c) { sa, sb ->
            host.symm(alpha, sa, symmetricStructure(lower), sb, beta, c, rightSide = right)
        }
    }

    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, DenseMatrixOperation.Trsm, workspace)

    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, DenseMatrixOperation.Trmm, workspace)

    private fun triangularMatrix(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        operation: DenseMatrixOperation,
        workspace: Workspace?,
    ) {
        val solve = operation == DenseMatrixOperation.Trsm
        val what = if (solve) "trsm" else "trmm"
        requireTriangularMatrixOperands(a, b, right, what)
        // This API reads no coefficients when alpha is zero; eligibility keeps that case on the
        // portable path so every host binding follows the same unused-operand contract.
        if (!host(operation, eligibleCall(operation, b.rows, b.cols, a.rows, alpha))) {
            if (solve) {
                portable.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            } else {
                portable.trmm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            }
            return
        }
        staged(workspace, a, a.values === b.values) { sa ->
            val structure = triangle(lower, unitDiag)
            if (solve) {
                host.trsm(alpha, sa, structure, transpose, b, rightSide = right)
            } else {
                host.trmm(alpha, sa, structure, transpose, b, rightSide = right)
            }
        }
    }
}

/** Symbol mapping and required call facts; performance eligibility belongs to each platform. */
internal object HostDenseCall {
    /**
     * The CBLAS entry point for a whole operation, or null for windowed GER.
     *
     * Windowed GER stays portable because no binding takes its windows, which offset both vectors as well as
     * the matrix.
     */
    fun entryPointFor(operation: DenseMatrixOperation): BlasOperation? = when (operation) {
        DenseMatrixOperation.Gemv, DenseMatrixOperation.GemvTransposed -> BlasOperation.Gemv
        DenseMatrixOperation.Symv -> BlasOperation.Symv
        DenseMatrixOperation.Ger -> BlasOperation.Ger
        DenseMatrixOperation.Syr -> BlasOperation.Syr
        DenseMatrixOperation.Syr2 -> BlasOperation.Syr2
        DenseMatrixOperation.Trmv, DenseMatrixOperation.TrmvTransposed -> BlasOperation.Trmv
        DenseMatrixOperation.Trsv, DenseMatrixOperation.TrsvTransposed -> BlasOperation.Trsv
        DenseMatrixOperation.Gemm -> BlasOperation.Gemm
        DenseMatrixOperation.Gemmt -> BlasOperation.Gemmt
        DenseMatrixOperation.Symm -> BlasOperation.Symm
        DenseMatrixOperation.Syrk -> BlasOperation.Syrk
        DenseMatrixOperation.Syr2k -> BlasOperation.Syr2k
        DenseMatrixOperation.Trmm -> BlasOperation.Trmm
        DenseMatrixOperation.Trsm -> BlasOperation.Trsm
        DenseMatrixOperation.GerWindow -> null
    }

    /**
     * Whether [operation] needs a shared dimension of its own before anything about it can be settled.
     *
     * The same rule the portable reporter holds to, and holding to it here is what keeps a host route from
     * answering a question the portable one refuses. A Level 3 call with no depth is not a small call: it is
     * a call whose extents have not been stated, and a route built from the other two would be a description
     * of a product nobody described.
     */
    fun needsDepth(operation: DenseMatrixOperation): Boolean = when (operation) {
        DenseMatrixOperation.Gemm,
        DenseMatrixOperation.Gemmt, DenseMatrixOperation.Symm,
        DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k, DenseMatrixOperation.Trmm,
        DenseMatrixOperation.Trsm,
        -> true

        else -> false
    }
}

/** The general route of one whole entry point, with no operands to settle a quick return from. */
private fun Blas.routeOf(operation: BlasOperation): CallRoute = routeOf(operation, emptyList(), emptyList())

/** A caller-owned array as the contiguous vector operand BLAS takes. */
internal fun DoubleArray.asVector(): DenseVector = DenseVector.wrap(this)
