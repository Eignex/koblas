@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
)

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.RouteKind
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.PortableDenseBlas
import com.eignex.koblas.internal.dense.triangularMatrix
import com.eignex.koblas.internal.operands.regionVectorOverlap
import com.eignex.koblas.internal.operands.sameRegion
import com.eignex.koblas.internal.operands.staged
import com.eignex.koblas.internal.operands.stagedRegion
import com.eignex.koblas.internal.operands.stagedVectors
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.CallRoute

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
        val depth = if (HostDenseCall.needsDepth(operation)) {
            call.depth ?: return portable.routeOf(operation, call)
        } else {
            1
        }
        val permitted = eligibleCall(
            operation,
            call.rows,
            call.columns,
            depth,
            call.alpha,
            call.borrowed,
        )
        if (!host(operation, permitted)) return portable.routeOf(operation, call)
        val entry = requireNotNull(HostDenseCall.entryPointFor(operation))
        // Work, whole-entry-point eligibility and exports are settled above; only platform transfer remains.
        val hostRoute = host.routeOf(entry)
        val components = ArrayList<String>(2)
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

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        val operation = if (transpose) DenseMatrixOperation.GemvTransposed else DenseMatrixOperation.Gemv
        if (host(operation, eligibleCall(operation, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix))) {
            gemv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), transpose, workspace)
        } else {
            portable.gemv(alpha, a, x, beta, y, transpose, workspace)
        }
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        val operation = if (transpose) DenseMatrixOperation.GemvTransposed else DenseMatrixOperation.Gemv
        if (!host(operation, eligibleCall(operation, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix))) {
            portable.gemv(alpha, a, x, beta, y, transpose, workspace)
            return
        }
        stagedRegion(workspace, a, y) { sa ->
            staged(workspace, x, x.overlaps(y)) { sx ->
                host.gemv(alpha, sa, transpose, sx, beta, y)
            }
        }
    }

    override fun transpose(a: DenseMatrixRegion): DenseMatrix = portable.transpose(a)

    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        val operation = DenseMatrixOperation.Symv
        if (host(operation, eligibleCall(operation, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix))) {
            symv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), lower, workspace)
        } else {
            portable.symv(alpha, a, x, beta, y, lower, workspace)
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (!host(
                DenseMatrixOperation.Symv,
                eligibleCall(DenseMatrixOperation.Symv, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix),
            )
        ) {
            portable.symv(alpha, a, x, beta, y, lower, workspace)
            return
        }
        stagedRegion(workspace, a, y, lower = lower) { sa ->
            staged(workspace, x, x.overlaps(y)) { sx ->
                host.symv(alpha, sa, symmetricStructure(lower), sx, beta, y)
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
    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrixRegion, workspace: Workspace?) {
        val operation = DenseMatrixOperation.Ger
        if (host(operation, eligibleCall(operation, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix))) {
            ger(alpha, DenseVector.wrap(x), DenseVector.wrap(y), a, workspace)
        } else {
            portable.ger(alpha, x, y, a, workspace)
        }
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, workspace: Workspace?) {
        requireGerOperands(x.size, y.size, a)
        if (!host(
                DenseMatrixOperation.Ger,
                eligibleCall(DenseMatrixOperation.Ger, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix),
            )
        ) {
            portable.ger(alpha, x, y, a, workspace)
            return
        }
        stagedVectors(workspace, x, y, a) { sx, sy ->
            host.ger(alpha, sx, sy, a)
        }
    }

    // Whole-matrix bindings do not expose window offsets or leading dimensions.
    @Suppress("LongParameterList") // the rank update and its three independently addressed windows
    override fun ger(
        alpha: Double,
        x: DoubleArray,
        y: DoubleArray,
        a: DenseMatrixRegion,
        rowOffset: Int,
        columnOffset: Int,
        rows: Int,
        columns: Int,
        xOffset: Int,
        yOffset: Int,
        workspace: Workspace?,
    ): Unit = portable.ger(alpha, x, y, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset, workspace)

    @Suppress("LongParameterList") // the BLAS dsyr signature plus the workspace
    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, lower: Boolean, workspace: Workspace?) {
        requireSyrOperands(a, x.size, "syr")
        if (!host(
                DenseMatrixOperation.Syr,
                eligibleCall(DenseMatrixOperation.Syr, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix),
            )
        ) {
            portable.syr(alpha, x, a, lower, workspace)
            return
        }
        staged(workspace, x, regionVectorOverlap(a, x)) { sx ->
            host.syr(alpha, sx, a, symmetricStructure(lower))
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2 signature plus the workspace
    override fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2Operands(a, x.size, y.size, "syr2")
        if (!host(
                DenseMatrixOperation.Syr2,
                eligibleCall(DenseMatrixOperation.Syr2, a.rows, a.cols, alpha = alpha, borrowed = a !is DenseMatrix),
            )
        ) {
            portable.syr2(alpha, x, y, a, lower, workspace)
            return
        }
        stagedVectors(workspace, x, y, a) { sx, sy ->
            host.syr2(alpha, sx, sy, a, symmetricStructure(lower))
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        val operation = if (transpose) DenseMatrixOperation.TrsvTransposed else DenseMatrixOperation.Trsv
        if (host(operation, eligibleCall(operation, a.rows, a.cols, borrowed = a !is DenseMatrix))) {
            trsv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        } else {
            portable.trsv(a, x, lower, transpose, unitDiag, workspace)
        }
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DenseVector,
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
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        val operation = if (transpose) DenseMatrixOperation.TrmvTransposed else DenseMatrixOperation.Trmv
        if (host(operation, eligibleCall(operation, a.rows, a.cols, borrowed = a !is DenseMatrix))) {
            trmv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        } else {
            portable.trmv(a, x, lower, transpose, unitDiag, workspace)
        }
    }

    override fun trmv(
        a: DenseMatrixRegion,
        x: DenseVector,
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
        a: DenseMatrixRegion,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        operation: DenseMatrixOperation,
        what: String,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, what)
        val solve = operation == DenseMatrixOperation.Trsv || operation == DenseMatrixOperation.TrsvTransposed
        if (!host(operation, eligibleCall(operation, a.rows, a.cols, borrowed = a !is DenseMatrix))) {
            if (solve) {
                portable.trsv(a, x, lower, transpose, unitDiag, workspace)
            } else {
                portable.trmv(a, x, lower, transpose, unitDiag, workspace)
            }
            return
        }
        // The triangle is the only operand that can share the destination, since x is the destination.
        stagedRegion(workspace, a, x, lower = lower, unitDiag = unitDiag) { sa ->
            val structure = triangle(lower, unitDiag)
            if (solve) {
                host.trsv(sa, structure, transpose, x)
            } else {
                host.trmv(sa, structure, transpose, x)
            }
        }
    }

    // Contiguous column-major operands need staging only when they overlap the destination.

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
        requireGemmOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        val gemm = DenseMatrixOperation.Gemm
        if (!host(
                gemm,
                eligibleCall(
                    gemm,
                    c.rows,
                    c.cols,
                    depth,
                    alpha,
                    a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
                ),
            )
        ) {
            portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                host.gemm(alpha, sa, transposeA, sb, transposeB, beta, c)
            }
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        val gemmt = DenseMatrixOperation.Gemmt
        if (!host(
                gemmt,
                eligibleCall(
                    gemmt,
                    c.rows,
                    c.cols,
                    depth,
                    alpha,
                    a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
                ),
            )
        ) {
            portable.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, workspace)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                host.gemmt(alpha, sa, transposeA, sb, transposeB, beta, c, symmetricStructure(lower))
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
        requireSyrkOperands(a, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        val syrk = DenseMatrixOperation.Syrk
        if (!host(syrk, eligibleCall(syrk, c.rows, c.cols, depth, alpha, a !is DenseMatrix || c !is DenseMatrix))) {
            portable.syrk(alpha, a, transpose, beta, c, lower, workspace)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            host.syrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2kOperands(a, b, transpose, c)
        val depth = if (transpose) a.rows else a.cols
        val syr2k = DenseMatrixOperation.Syr2k
        if (!host(
                syr2k,
                eligibleCall(
                    syr2k,
                    c.rows,
                    c.cols,
                    depth,
                    alpha,
                    a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
                ),
            )
        ) {
            portable.syr2k(alpha, a, b, transpose, beta, c, lower, workspace)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                host.syr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
            }
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        val symm = DenseMatrixOperation.Symm
        if (!host(
                symm,
                eligibleCall(
                    symm,
                    c.rows,
                    c.cols,
                    a.rows,
                    alpha,
                    a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix,
                ),
            )
        ) {
            portable.symm(alpha, a, b, beta, c, lower, right, workspace)
            return
        }
        stagedRegion(workspace, a, c, lower = if (sameRegion(a, b)) null else lower) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                host.symm(alpha, sa, symmetricStructure(lower), sb, beta, c, rightSide = right)
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
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, DenseMatrixOperation.Trsm, workspace)

    override fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, DenseMatrixOperation.Trmm, workspace)

    private fun triangularMatrix(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
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
        if (!host(
                operation,
                eligibleCall(operation, b.rows, b.cols, a.rows, alpha, a !is DenseMatrix || b !is DenseMatrix),
            )
        ) {
            if (solve) {
                portable.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            } else {
                portable.trmm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            }
            return
        }
        stagedRegion(workspace, a, b, lower = lower, unitDiag = unitDiag) { sa ->
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
