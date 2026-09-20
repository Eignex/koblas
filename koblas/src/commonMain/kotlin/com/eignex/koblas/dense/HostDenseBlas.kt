@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
)

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.StridedVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.RouteKind
import com.eignex.koblas.vendor.requireGemmOperands
import com.eignex.koblas.vendor.requireGemmtOperands
import com.eignex.koblas.vendor.requireGemvOperands
import com.eignex.koblas.vendor.requireGerOperands
import com.eignex.koblas.vendor.requireSymmOperands
import com.eignex.koblas.vendor.requireSymvOperands
import com.eignex.koblas.vendor.requireSyr2Operands
import com.eignex.koblas.vendor.requireSyr2kOperands
import com.eignex.koblas.vendor.requireSyrOperands
import com.eignex.koblas.vendor.requireSyrkOperands
import com.eignex.koblas.vendor.requireTriangularMatrixOperands
import com.eignex.koblas.vendor.requireTriangularVectorOperands

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

/**
 * Dense Level 2 and 3 served by an installed host library where one is eligible, and portably where it is not.
 *
 * The composition a platform default takes when the host has a tuned library and this library's own Kotlin
 * arithmetic is not vectorised: Kotlin/Native today. It is not the explicit host seam. [VendorDenseBlas] is
 * that, and the difference is the point of both existing. The explicit seam raises where no library is
 * installed and rejects an operand that shares the destination, because a benchmark arm asking for a vendor
 * measurement must not be handed something else; this one never raises for those reasons, because it is what
 * an ordinary call gets and an ordinary call is owed an answer.
 *
 * Six things decide a call, in this order, all of them before anything is written:
 *
 *  1. The operation has a CBLAS entry point of its own. A product between operands packed for this library's
 *     own register tile does not: the layout is ours and a library cannot read it, so those calls stay on the
 *     portable schedule and their routes say so.
 *  2. This library exports that entry point. `gemmt` is the one a supported vendor may legitimately lack, and
 *     composing it from a full product plus a triangle copy is work this library would be doing anyway.
 *  3. What [DenseBlas] promises for this routine and this multiplier is something a library also gives.
 *     [HostDensePolicy.structurallyCompatible] and [HostDensePolicy.multiplierIsCompatible] are the whole of
 *     that question, and both are answered before a write rather than discovered from a result.
 *  4. The call states the shared dimension its operation is defined over. A Level 3 route with none is not a
 *     small call but an unstated one, and the portable path is where that is refused rather than answered.
 *  5. The call does arithmetic at all. A zero multiplier, an empty extent or an empty shared dimension is a
 *     call whose no-read rules this library states and a vendor is not held to, so the portable path keeps
 *     them rather than the policy having to reason about what a given library does with them.
 *  6. There is enough arithmetic to pay for reaching the library. See [HostDensePolicy.MINIMUM_WORK].
 *
 * A call that gets here is one where the library's freedom cannot be told apart from what [DenseBlas]
 * promises, so what comes back is what an ordinary call is owed. Within that, the accumulation order is the
 * library's, which is latitude [DenseBlas.gemm] already states for any built-in schedule. The rest of the
 * contract is unchanged and belongs to this layer: shapes are validated before anything is written, a zero
 * multiplier reads no operand, an unselected triangle is neither read nor written, and an input that shares
 * the destination's buffer is staged into scratch first. A whole-call binding cannot take that overlap, so
 * the copy is made here and lent by [Workspace] like any other.
 *
 * Immutable and safe to share. A call takes scratch from the workspace it was given and nothing else, so
 * independent calls with distinct workspaces and distinct destinations do not interact; the binding underneath
 * is documented to hold the same.
 */
internal class HostDenseBlas(
    private val portable: PortableDenseBlas,
    private val host: Blas,
    private val minimumWork: Long = HostDensePolicy.MINIMUM_WORK,
) : RoutedDenseBlas {

    /**
     * Whether this call goes to the library, asked before it writes anything.
     *
     * [work] is what [HostDensePolicy.multiplyAdds] made of the call's own facts, and null there is not a
     * small call: it is an operation with no entry point, a routine or a multiplier whose documented
     * behaviour a library need not share, a call whose contract stops before the arithmetic, or one that
     * never stated the shared dimension it is defined over. All of them stay portable, and the route agrees
     * because it asks this question of the same facts.
     */
    private fun host(operation: DenseMatrixOperation, work: Long?): Boolean {
        if (work == null) return false
        val entry = HostDensePolicy.entryPointFor(operation) ?: return false
        if (entry !in host.directlyImplemented) return false
        return work >= minimumWork
    }

    /** The multiply-adds of a concrete call, over the extents [DenseCall] names for its operation. */
    private fun work(
        operation: DenseMatrixOperation,
        rows: Int,
        columns: Int,
        depth: Int? = null,
        alpha: Double = 1.0,
        right: Boolean = false,
    ): Long? = HostDensePolicy.multiplyAdds(operation, rows, columns, depth, alpha, right)

    override fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute {
        val work = HostDensePolicy.multiplyAdds(operation, call)
        if (!host(operation, work)) return portable.routeOf(operation, call)
        val entry = requireNotNull(HostDensePolicy.entryPointFor(operation))
        // Asked with no operands, which is the general answer, and sound only because the three questions the
        // operand lists exist for have already been settled above: this call does work, it is one whole entry
        // point, and the library exports it. What the binding still contributes is its own platform transfer.
        val hostRoute = host.routeOf(entry)
        val components = ArrayList<String>(2)
        if (call.aliased) components.add(HOST_STAGING)
        components.add("${host.vendor.vendorName.lowercase()}/${entry.entryPoint}")
        return DenseMatrixRoute(
            operation = operation,
            kind = RouteKind.Direct,
            scheduling = HOST_SCHEDULING,
            entryPoint = operation.entryPoint,
            components = components,
            executionGroup = 0,
            reason = "the whole call is one ${host.vendor.vendorName} entry point at ${host.libraryPath}, " +
                "version ${host.version}, ${host.threadEvidence.label}" +
                if (call.aliased) "; an input sharing the destination is copied into scratch first" else "",
            host = hostRoute,
        )
    }

    // Level 2. One matrix operand, whose extents are the whole of the arithmetic.

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
        if (!host(operation, work(operation, a.rows, a.cols, alpha = alpha))) {
            portable.gemv(alpha, a, x, beta, y, transpose, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === y) { sa ->
            stagedArray(workspace, x, x === y) { sx ->
                host.gemv(alpha, sa, transpose, sx.asVector(), beta, y.asVector())
            }
        }
    }

    override fun transpose(a: DenseMatrix): DenseMatrix = portable.transpose(a)

    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSymvOperands(a, symmetricStructure(lower), x.size, y.size)
        if (!host(DenseMatrixOperation.Symv, work(DenseMatrixOperation.Symv, a.rows, a.cols, alpha = alpha))) {
            portable.symv(alpha, a, x, beta, y, lower)
            return
        }
        stagedMatrix(null, a, a.values === y) { sa ->
            stagedArray(null, x, x === y) { sx ->
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
    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireGerOperands(x.size, y.size, a)
        if (!host(DenseMatrixOperation.Ger, work(DenseMatrixOperation.Ger, a.rows, a.cols, alpha = alpha))) {
            portable.ger(alpha, x, y, a)
            return
        }
        stagedArray(null, x, x === a.values) { sx ->
            stagedArray(null, y, y === a.values) { sy ->
                host.ger(alpha, sx.asVector(), sy.asVector(), a)
            }
        }
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyrOperands(a, symmetricStructure(lower), "syr", x)
        if (!host(DenseMatrixOperation.Syr, work(DenseMatrixOperation.Syr, a.rows, a.cols, alpha = alpha))) {
            portable.syr(alpha, x, a, lower)
            return
        }
        stagedVector(null, x, x.values === a.values) { sx ->
            host.syr(alpha, sx, a, symmetricStructure(lower))
        }
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyr2Operands(a, symmetricStructure(lower), "syr2", x, y)
        if (!host(DenseMatrixOperation.Syr2, work(DenseMatrixOperation.Syr2, a.rows, a.cols, alpha = alpha))) {
            portable.syr2(alpha, x, y, a, lower)
            return
        }
        stagedVector(null, x, x.values === a.values) { sx ->
            stagedVector(null, y, y.values === a.values) { sy ->
                host.syr2(alpha, sx, sy, a, symmetricStructure(lower))
            }
        }
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        val operation = if (transpose) DenseMatrixOperation.TrsvTransposed else DenseMatrixOperation.Trsv
        triangularVector(a, x, lower, transpose, unitDiag, operation, "trsv")
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        val operation = if (transpose) DenseMatrixOperation.TrmvTransposed else DenseMatrixOperation.Trmv
        triangularVector(a, x, lower, transpose, unitDiag, operation, "trmv")
    }

    private fun triangularVector(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        operation: DenseMatrixOperation,
        what: String,
    ) {
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.size, what)
        val solve = operation == DenseMatrixOperation.Trsv || operation == DenseMatrixOperation.TrsvTransposed
        if (!host(operation, work(operation, a.rows, a.cols))) {
            if (solve) {
                portable.trsv(
                    a,
                    x,
                    lower,
                    transpose,
                    unitDiag,
                )
            } else {
                portable.trmv(a, x, lower, transpose, unitDiag)
            }
            return
        }
        // The triangle is the only operand that can share the destination, since x is the destination.
        stagedMatrix(null, a, a.values === x) { sa ->
            val structure = triangle(lower, unitDiag)
            if (solve) {
                host.trsv(sa, structure, transpose, x.asVector())
            } else {
                host.trmv(sa, structure, transpose, x.asVector())
            }
        }
    }

    // Level 3. Every operand is contiguous column-major, so only an overlap with the destination is staged.

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
        if (!host(gemm, work(gemm, c.rows, c.cols, depth, alpha))) {
            portable.gemm(alpha, a, transposeA, b, transposeB, beta, c, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === c.values) { sa ->
            stagedMatrix(workspace, b, b.values === c.values) { sb ->
                host.gemm(alpha, sa, transposeA, sb, transposeB, beta, c)
            }
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
        requireGemmtOperands(a, transposeA, b, transposeB, c, symmetricStructure(lower))
        val depth = if (transposeA) a.rows else a.cols
        val gemmt = DenseMatrixOperation.Gemmt
        if (!host(gemmt, work(gemmt, c.rows, c.cols, depth, alpha))) {
            portable.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === c.values) { sa ->
            stagedMatrix(workspace, b, b.values === c.values) { sb ->
                host.gemmt(alpha, sa, transposeA, sb, transposeB, beta, c, symmetricStructure(lower))
            }
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
        requireSyrkOperands(a, transpose, c, symmetricStructure(lower))
        val depth = if (transpose) a.rows else a.cols
        val syrk = DenseMatrixOperation.Syrk
        if (!host(syrk, work(syrk, c.rows, c.cols, depth, alpha))) {
            portable.syrk(alpha, a, transpose, beta, c, lower, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === c.values) { sa ->
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
        requireSyr2kOperands(a, b, transpose, c, symmetricStructure(lower))
        val depth = if (transpose) a.rows else a.cols
        val syr2k = DenseMatrixOperation.Syr2k
        if (!host(syr2k, work(syr2k, c.rows, c.cols, depth, alpha))) {
            portable.syr2k(alpha, a, b, transpose, beta, c, lower, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === c.values) { sa ->
            stagedMatrix(workspace, b, b.values === c.values) { sb ->
                host.syr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
            }
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
        requireSymmOperands(a, symmetricStructure(lower), b, c, right)
        val symm = DenseMatrixOperation.Symm
        if (!host(symm, work(symm, c.rows, c.cols, a.rows, alpha))) {
            portable.symm(alpha, a, b, beta, c, lower, right, workspace)
            return
        }
        stagedMatrix(workspace, a, a.values === c.values) { sa ->
            stagedMatrix(workspace, b, b.values === c.values) { sb ->
                host.symm(alpha, sa, symmetricStructure(lower), sb, beta, c, rightSide = right)
            }
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
        requireTriangularMatrixOperands(a, triangle(lower, unitDiag), b, right, what)
        // A zero alpha zeroes the right-hand sides and reads no coefficient, which this library states and a
        // library need not; [HostDensePolicy] answers null for it and the portable path keeps it.
        if (!host(operation, work(operation, b.rows, b.cols, a.rows, alpha, right))) {
            if (solve) {
                portable.trsm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            } else {
                portable.trmm(a, b, lower, transpose, unitDiag, right, alpha, workspace)
            }
            return
        }
        stagedMatrix(workspace, a, a.values === b.values) { sa ->
            val structure = triangle(lower, unitDiag)
            if (solve) {
                host.trsm(alpha, sa, structure, transpose, b, rightSide = right)
            } else {
                host.trmm(alpha, sa, structure, transpose, b, rightSide = right)
            }
        }
    }

    /**
     * [block] with a copy of [a] when [aliased], and with [a] itself when it is not.
     *
     * The loan lasts the call, which is exactly how long the library reads it for. A whole-call binding takes
     * the caller's storage as it lies and has no way to be told that two of its operands are one buffer, so
     * the separation is made here rather than asked of the library.
     */
    private inline fun <T> stagedMatrix(
        workspace: Workspace?,
        a: DenseMatrix,
        aliased: Boolean,
        block: (DenseMatrix) -> T,
    ): T {
        if (!aliased) return block(a)
        return workspace.borrow(a.values.size) { copy ->
            a.values.copyInto(copy)
            block(DenseMatrix.wrap(a.rows, a.cols, copy))
        }
    }

    /** [stagedMatrix] for an operand a caller handed over as a plain array. */
    private inline fun <T> stagedArray(
        workspace: Workspace?,
        values: DoubleArray,
        aliased: Boolean,
        block: (DoubleArray) -> T,
    ): T {
        if (!aliased) return block(values)
        return workspace.borrow(values.size) { copy ->
            values.copyInto(copy)
            block(copy)
        }
    }

    /** [stagedMatrix] for a vector operand, whose origin and spacing the copy keeps. */
    private inline fun <T> stagedVector(
        workspace: Workspace?,
        x: DenseVector,
        aliased: Boolean,
        block: (DenseVector) -> T,
    ): T {
        if (!aliased) return block(x)
        return workspace.borrow(x.values.size) { copy ->
            x.values.copyInto(copy)
            block(StridedVector(copy, x.offset, x.size, x.stride))
        }
    }
}

/**
 * When a whole dense call is handed to an installed library rather than run on the portable schedule.
 *
 * A conservative fixed policy rather than a tuned one. Three questions settle it. Is this operation one whole
 * CBLAS entry point that the library exports, and does the call carry the facts that settle its own shape;
 * can a library's own freedom be told apart from what [DenseBlas] promises for this call, in which case the
 * promise decides and the call stays here; and has the arithmetic grown to a size where a foreign call and a
 * pin per operand are a small part of it.
 *
 * The middle question is the one that is not about speed. A built-in call keeps the behaviour [DenseBlas]
 * states for it on every platform, and a library is installed by the host rather than chosen by the caller,
 * so an ordinary call cannot acquire new latitude from one being present. Where compatibility is not
 * something this policy can establish, the portable schedule is chosen, and it is chosen before anything is
 * written. Callers who want a library's own answers ask for one through [com.eignex.koblas.vendor.Blas],
 * which documents that latitude and is reached deliberately.
 *
 * One size threshold for all of them rather than one per operation. The eight Level 1 break-evens that the
 * Kotlin/Native vector kernels split differ because a per-call cost of a few tens of nanoseconds is most of
 * the work at those widths. A Level 2 or 3 call is quadratic or cubic in its extents, so it crosses that cost
 * within a few steps of the extents either way, and a second constant would be distinguishing calls that are
 * both far on one side of the line.
 */
internal object HostDensePolicy {
    /**
     * The multiply-add count from which a whole call is handed over.
     *
     * Deliberately on the late side of where the two sides meet, because the errors are not symmetric:
     * handing a call over too early makes it slower than the loop it replaced, and holding one back only
     * forgoes a win. It is a fixed conservative policy and not a tuned number.
     *
     * The size was chosen from a bounded local ladder of `gemm`, `gemv` and `trsm` on one machine, comparing
     * this library's exact portable Native arm against the explicit oneMKL and OpenBLAS ones through
     * `koblas-bench`. What those rows establish is roughly where the arithmetic grows past the cost of
     * reaching a library, which on that host was in the hundreds of multiply-adds. Their limits are as
     * important: they are an explicit whole-call binding rather than this composition, so they do not include
     * the validation, eligibility and operand wrapping around it; they are one CPU, two libraries and one
     * Kotlin target; and they were taken on a shared desktop. So this is not a speedup figure and not a
     * crossing anyone should assume holds elsewhere. Another host calls for that ladder again rather than an
     * adjustment to this.
     */
    const val MINIMUM_WORK: Long = 1024L

    /**
     * The CBLAS entry point [operation] is one whole call to, or null where it is not one at all.
     *
     * The packed products are the operations with no entry point. Their operands are grouped for this
     * library's own register tile, which is a layout of ours and not one a library has an argument for, so a
     * call over a retained panel stays on the portable schedule however large it is. An engine holding a
     * vendor binding is not evidence that such a call reached it, which is what keeps the route honest.
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

        DenseMatrixOperation.GemmPacked,
        DenseMatrixOperation.GemmPackedLeft,
        DenseMatrixOperation.GemmPackedRight,
        -> null
    }

    /**
     * Whether a library may serve [operation] at all without changing what [DenseBlas] promises for it.
     *
     * One entry, and it is [DenseMatrixOperation.Syr2k]. That routine is documented as the two products it
     * is defined as, composed rather than fused, so its result is `alpha · s₁ + alpha · s₂` where each sum
     * is accumulated on its own. A library's `dsyr2k` may legitimately walk both at once and add the pair of
     * terms before accumulating, and the two are not the same function of finite operands: with two rows and
     * a shared dimension whose columns are `[1, -1]` against a second operand of halved maxima, one of the
     * separate sums overflows to an infinity and the other to its negative, so composing gives a NaN where
     * an interleaved traversal cancels each pair and gives zero. Nothing about the operands or the scalars
     * distinguishes the two cheaply, so this routine is not handed over.
     *
     * The other thirteen have no such statement of their own, and what they do promise about the placement
     * of a multiplier is kept by [multiplierIsCompatible] instead.
     */
    fun structurallyCompatible(operation: DenseMatrixOperation): Boolean = operation != DenseMatrixOperation.Syr2k

    /**
     * The routines whose result [DenseBlas] states as a multiplier applied to an accumulated sum.
     *
     * [DenseBlas.gemm] says it, [DenseBlas.syrk] says it scales its one product exactly as `gemm` describes,
     * and `gemmt` is that product restricted to a triangle. The rest of the bound surface states no placement
     * at all, so a library's own is the selected implementation's answer there, which is what this library
     * says about everything the standard leaves open.
     */
    private val SCALES_AN_ACCUMULATED_SUM = setOf(
        DenseMatrixOperation.Gemm,
        DenseMatrixOperation.Gemmt,
        DenseMatrixOperation.Syrk,
    )

    /**
     * Whether a library may serve a call carrying this [alpha] without changing what [DenseBlas] promises.
     *
     * For the routines in [SCALES_AN_ACCUMULATED_SUM] the rule is that there is no multiplier to place: the
     * multiplier is one. Nothing weaker survives, and the reason is that a library may legally scale an
     * operand before multiplying, which reference BLAS does in places. Over a shared dimension of one there
     * is no partition of a sum to explain the difference, and two finite fixtures settle it. With a
     * multiplier of `Double.MAX_VALUE` against an operand entry of zero and another of two, this library's
     * `(0 · 2) · alpha` is zero while pre-scaling the two gives an infinity and then a NaN. With a
     * multiplier of a half against `Double.MAX_VALUE` and two, this library's `(MAX · 2) · alpha` overflows
     * to an infinity while pre-scaling the two leaves `MAX` finite. Both are ordinary finite arguments, so
     * neither a finiteness test nor the repartitioning [DenseBlas.gemm] already allows covers them.
     *
     * A unit multiplier does: with nothing to scale, where the scaling would have gone cannot be observed.
     * The cost is that a scaled product of those three routines keeps this library's schedule, which is a
     * fallback the plan permits and which happens before anything is written.
     *
     * Every other bound routine is unrestricted here, because [DenseBlas] promises nothing about where their
     * multiplier lands and the route of the call names the library that answered.
     */
    fun multiplierIsCompatible(operation: DenseMatrixOperation, alpha: Double): Boolean =
        alpha == 1.0 || operation !in SCALES_AN_ACCUMULATED_SUM

    /**
     * Whether [operation] needs a shared dimension of its own before anything about it can be settled.
     *
     * The same rule the portable reporter holds to, and holding to it here is what keeps a host route from
     * answering a question the portable one refuses. A Level 3 call with no depth is not a small call: it is
     * a call whose extents have not been stated, and a route built from the other two would be a description
     * of a product nobody described.
     */
    fun needsDepth(operation: DenseMatrixOperation): Boolean = when (operation) {
        DenseMatrixOperation.Gemm, DenseMatrixOperation.GemmPacked, DenseMatrixOperation.GemmPackedLeft,
        DenseMatrixOperation.GemmPackedRight, DenseMatrixOperation.Gemmt, DenseMatrixOperation.Symm,
        DenseMatrixOperation.Syrk, DenseMatrixOperation.Syr2k, DenseMatrixOperation.Trmm,
        DenseMatrixOperation.Trsm,
        -> true

        else -> false
    }

    /**
     * The multiply-adds a call may hand to a library, or null where it may not hand it any.
     *
     * Null covers five things that have one answer. An operation with no entry point of its own; a routine
     * whose documented behaviour a library need not share, which is [structurallyCompatible]; a multiplier
     * this routine documents the placement of, which is [multiplierIsCompatible]; a call whose own contract
     * stops before the arithmetic; and a call that did not state the shared dimension its operation is
     * defined over. Each of them belongs on the portable schedule, and the reason differs: the
     * first has nowhere else to go, the second and third are promises this library made, the fourth turns on
     * no-read rules the standard leaves open, and the fifth is refused rather than answered.
     *
     * The extents are [DenseCall]'s, so the execution paths and [HostDenseBlas.routeOf] pass the same facts
     * and cannot disagree about which of them the call has.
     */
    @Suppress("LongParameterList", "CyclomaticComplexMethod") // the extents a call is measured over
    fun multiplyAdds(
        operation: DenseMatrixOperation,
        rows: Int,
        columns: Int,
        depth: Int?,
        alpha: Double,
        right: Boolean,
    ): Long? {
        if (entryPointFor(operation) == null) return null
        if (!structurallyCompatible(operation) || !multiplierIsCompatible(operation, alpha)) return null
        if (needsDepth(operation) && depth == null) return null
        // The rule the portable reporter stops on, restated over the same three extents.
        if (alpha == 0.0 || rows == 0 || columns == 0 || depth == 0) return null
        val m = rows.toLong()
        val n = columns.toLong()
        val k = (depth ?: 1).toLong()
        return when (operation) {
            DenseMatrixOperation.Gemv, DenseMatrixOperation.GemvTransposed,
            DenseMatrixOperation.Symv, DenseMatrixOperation.Ger, DenseMatrixOperation.Syr2,
            -> m * n

            DenseMatrixOperation.Syr,
            DenseMatrixOperation.Trmv, DenseMatrixOperation.TrmvTransposed,
            DenseMatrixOperation.Trsv, DenseMatrixOperation.TrsvTransposed,
            -> m * n / 2

            DenseMatrixOperation.Gemm, DenseMatrixOperation.Symm, DenseMatrixOperation.Syr2k -> m * n * k

            DenseMatrixOperation.Gemmt, DenseMatrixOperation.Syrk -> m * n * k / 2

            // The triangle's order is the depth, and the right-hand sides are B's other extent.
            DenseMatrixOperation.Trmm, DenseMatrixOperation.Trsm -> k * k * (if (right) m else n) / 2

            else -> null
        }
    }

    /** [multiplyAdds] over the extents a caller already holds as a [DenseCall]. */
    fun multiplyAdds(operation: DenseMatrixOperation, call: DenseCall): Long? =
        multiplyAdds(operation, call.rows, call.columns, call.depth, call.alpha, call.right)
}

/** The general route of one whole entry point, with no operands to settle a quick return from. */
private fun Blas.routeOf(operation: BlasOperation): CallRoute = routeOf(operation, emptyList(), emptyList())
