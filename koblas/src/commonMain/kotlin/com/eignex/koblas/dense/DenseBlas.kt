@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * Dense matrix routines implemented portably by built-in engines and optionally by explicit host bindings.
 *
 * Built-in calls validate shapes, preserve zero-multiplier no-read rules and snapshot permitted aliases.
 * Explicit vendor bindings retain their library's exceptional and accumulation behavior.
 *
 * A platform default may hand a whole Level 2 or 3 call to an installed library where one is available and
 * the call is large enough to pay for reaching it; Kotlin/Native's does, and the JVM's does not. Everything
 * stated here holds either way. A library is part of the host rather than something a caller asked for, so
 * an ordinary call does not acquire new behaviour from one being installed: where a routine's documented
 * result and a library's freedom can be told apart, the portable schedule is chosen, and it is chosen before
 * anything is written. What is left over is the accumulation order, which every built-in schedule is already
 * free to choose below. [com.eignex.koblas.KoblasEngine.denseRouteOf] says which of the two a given call
 * took, and reports the library and symbol where it was the host.
 *
 * An explicit host binding is the other thing, and is where a library's own answers are asked for
 * deliberately: [com.eignex.koblas.vendor.Blas] keeps that latitude and documents it.
 *
 * These are Koblas's own shapes for the same routines: array operands, booleans for the transpose and the
 * stored triangle, and overloads that allocate a result. [com.eignex.koblas.vendor.Blas] is the standard's own
 * shape underneath, and the transpose and structure travel beside an operand as flags there rather than being
 * inferred from it.
 *
 * Portable calls run on the caller's thread and need no native access. On the JVM an explicit host call copies
 * operands into native memory; Kotlin/Native pins caller storage for that binding.
 */
public interface DenseBlas {
    /**
     * `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose].
     *
     * [workspace] lends the staging a built-in call takes when [a] or [x] shares [y]. A call that needs no
     * staging takes nothing from it.
     */
    @Suppress("LongParameterList") // the BLAS dgemv signature plus the workspace
    public fun gemv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    )

    /** [gemv] with `alpha = 1, beta = 0`, into a fresh result. */
    public fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val y = DoubleArray(if (transpose) a.cols else a.rows)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /**
     * Fresh transposed [a]. For a product prefer the transpose flags on [gemv] and [gemm], which avoid a
     * caller-visible intermediate; this is for a caller that means to hold the transpose.
     *
     * A storage transform rather than arithmetic, so it stays a Kotlin loop: the standard has no entry point
     * for it, and the BLAS-like `omatcopy` extension is not one every supported library exports.
     */
    public fun transpose(a: DenseMatrix): DenseMatrix

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *
     * Built-in implementations snapshot an input that shares [c]; explicit vendor seams reject that overlap.
     * [workspace] lends that snapshot, the packed panels a blocked product copies its operands into, and the
     * accumulating column an unpacked one uses. A zero [alpha] or an empty shared dimension scales [c] and
     * reads no operand; a zero [beta] overwrites [c] without reading it, whatever stands there.
     *
     * In a built-in implementation [alpha] multiplies an accumulated sum of products, never an individual
     * entry of an operand. Which partition of the shared dimension is summed before that multiplication is
     * the schedule's: a product small enough to run where its operands lie sums the whole of it, and a
     * blocked one sums a block of it at a time and adds the scaled results. For ordinary finite operands
     * the difference between those is reassociation, of the same kind a regrouped sum always brings; where
     * a partial sum overflows or cancels it can be larger than that, and with an infinite [alpha] it is
     * categorical, since `alpha · (s₁ + s₂)` and `alpha · s₁ + alpha · s₂` need not agree and a call's
     * extents decide which it gets. A sum that comes to zero against an infinite [alpha] is a NaN either
     * way; what the placement rules out is a single zero *entry* of [a] or [b] producing one on its own.
     *
     * An explicit host binding is not held to that placement: a library is free to scale coefficients as it
     * goes, and [com.eignex.koblas.vendor.Blas] keeps its own latitude where the standard leaves this open.
     * That is why a default which composes an installed library keeps this call for itself whenever [alpha]
     * is not finite, which is the only way the placement is observable.
     */
    @Suppress("LongParameterList") // the BLAS dgemm signature
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    )

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed
     * as `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read
     * nor written. Built-in implementations snapshot input overlap with [c].
     *
     * Not every library exports it. Where one does not, the call is composed from `gemm` plus a triangle copy,
     * and the route of the call reports which of the two ran rather than leaving the name to imply the first.
     */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    public fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /** [gemm] with `alpha = 1, beta = 0`, into a fresh matrix. `A.cols` must equal `B.rows`. */
    public fun gemm(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     * Only the [lower] or upper triangle is written. Built-in implementations snapshot overlap with [c].
     *
     * One product, so [alpha] multiplies an accumulated sum exactly as [gemm] describes, with the same
     * latitude over which partition of the shared dimension is summed before that multiplication.
     */
    @Suppress("LongParameterList") // the BLAS dsyrk signature
    public fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /** `y = alpha · A · x + beta · y` for a symmetric [a] (BLAS `dsymv`). Only the [lower] triangle is read,
     *  diagonal included. */
    @Suppress("LongParameterList") // the BLAS dsymv signature
    public fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
    )

    /** `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     *  [lower] triangle of [a] is read; built-in implementations snapshot input overlap with [c]. */
    @Suppress("LongParameterList") // the BLAS dsymm signature
    public fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        right: Boolean = false,
        workspace: Workspace? = null,
    )

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`). */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix)

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. [x] must be dense or
     *  strided storage, which is what a vendor can address. */
    public fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean = true)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. Operand
     *  storage follows [syr]. */
    public fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean = true)

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Writes only the [lower] or upper triangle and snapshots input overlap in built-in engines.
     *
     * A built-in implementation composes the two products rather than fusing one traversal over both, so
     * [alpha] multiplies each of the two accumulated sums separately and the scaled results are added:
     * `alpha · s₁ + alpha · s₂` rather than `alpha · (s₁ + s₂)`. For ordinary finite operands that is a
     * reassociation of the kind [gemm] already describes. Where it is more than that it is categorical: with
     * an infinite [alpha] and one of the two sums coming to zero, this gives a NaN where a fused traversal
     * would give an infinity. The composition is what lets each half be packed, blocked and tiled like any
     * other product, and it is the semantics this library promises rather than an accident of scheduling.
     *
     * An explicit host binding is not held to it, for the reason [gemm] gives: a library's `dsyr2k` may well
     * fuse the two traversals, and finite operands are enough to tell the two apart, since one of the
     * separate sums can overflow to an infinity and the other to its negative where the fused pair cancels.
     * A default which composes an installed library therefore never composes one for this routine.
     */
    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    public fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a],
     * `op` transposing when [transpose] and [unitDiag] taking the diagonal as 1. [x] carries b in and x out.
     *
     * The diagonal is divided by, not tested: `dtrsv` carries no `info` and reports nothing, so a singular
     * triangle yields infinities or NaNs and the caller who needs the distinction tests the diagonal first.
     */
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags
     *  follow [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right. */
    @Suppress("LongParameterList") // the BLAS dtrsm signature
    public fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    )

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv]. */
    public fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
    )

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of
     *  [trsm]. */
    @Suppress("LongParameterList") // the BLAS dtrmm signature
    public fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    )
}
