@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * Dense matrix routines implemented portably by built-in engines and optionally by explicit host bindings.
 *
 * Built-in calls validate shapes, preserve zero-multiplier no-read rules and snapshot permitted aliases.
 * Explicit vendor bindings retain their library's exceptional and accumulation behavior.
 *
 * Every routine that can allocate scratch takes a [Workspace] to lend it, whether that scratch is an alias
 * snapshot, a gathered operand or the panels a blocked product packs. A call that needs none takes nothing
 * from the one it is handed, so passing one costs a routine that does not use it nothing.
 *
 * A platform default may hand a whole Level 2 or 3 call to an installed library where one is available and
 * the call is large enough to pay for reaching it. Kotlin/Native composes available hosts; the JVM may
 * compose oneMKL for eligible Level 3 calls with native access enabled. Everything
 * stated here holds either way. What is deliberately not stated is the accumulation order and where a
 * multiplier lands, which every built-in schedule already chooses for itself and a library chooses too.
 * [com.eignex.koblas.KoblasEngine.routeOf] says which of the two a given call took, and reports the library
 * and symbol where it was the host.
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
    /** [gemv] over a region with unit alpha and zero beta, into a fresh vector. */
    public fun gemv(a: DenseMatrixRegion, x: DoubleArray, transpose: Boolean = false): DoubleArray {
        val size = if (transpose) a.cols else a.rows
        requireGemvOperands(a, transpose, x.size, size)
        val y = DoubleArray(size)
        gemv(1.0, a, x, 0.0, y, transpose)
        return y
    }

    /** [gemm] over regions with unit alpha and zero beta, into a fresh contiguous matrix. */
    public fun gemm(a: DenseMatrixRegion, b: DenseMatrixRegion): DenseMatrix {
        requireProductOperands(a, false, b, false, "gemm")
        val c = DenseMatrix.zero(a.rows, b.cols)
        gemm(1.0, a, false, b, false, 0.0, c)
        return c
    }

    /**
     * Region overload of [gemv]. Portable engines address borrowed storage directly. Other implementations
     * pack regions into contiguous workspace loans before calling their contiguous overloads. Host-composed
     * engines use the portable region path for a call the host is not eligible for. An eligible call reaches
     * the library in place where its binding takes a region's offset and leading dimension, as on Kotlin/Native,
     * and is packed otherwise.
     * Overlapping inputs retain their pre-call entries. Entries outside a destination region are untouched;
     * the transpose, triangle and zero-multiplier rules of the contiguous overload also apply to regions.
     * Packing, alias snapshots and ordinary backend panel scratch mean a view is not an allocation guarantee.
     * For route queries set [DenseCall.borrowed], and supply the documented leading dimensions where padded.
     */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): Unit =
        regionGemv(this, alpha, a, x, beta, y, transpose, workspace)

    /** Region overload of [gemm]; storage addressing and backend packing follow the region [gemv] overload. */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace? = null,
    ): Unit =
        regionGemm(this, alpha, a, transposeA, b, transposeB, beta, c, workspace)

    /** Region overload of [syrk]; storage addressing and backend packing follow the region [gemv] overload. */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        workspace: Workspace? = null,
    ): Unit =
        regionSyrk(this, alpha, a, transpose, beta, c, lower, workspace)

    /** Region overload of [trsv]; storage addressing and backend packing follow the region [gemv] overload. */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        workspace: Workspace? = null,
    ): Unit =
        regionTrsv(this, a, x, lower, transpose, unitDiag, workspace)

    /** Region overload of [trsm]; storage addressing and backend packing follow the region [gemv] overload. */
    @Suppress("LongParameterList") // BLAS operands, flags and workspace
    public fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        workspace: Workspace? = null,
    ): Unit =
        regionTrsm(this, a, b, lower, transpose, unitDiag, right, alpha, workspace)

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
        val size = if (transpose) a.cols else a.rows
        requireGemvOperands(a, transpose, x.size, size)
        val y = DoubleArray(size)
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
     * [workspace] lends that snapshot, packed panels for a blocked product, an accumulating column for an
     * unpacked product, and a coefficient column for a gathered strided reduction. A zero [alpha] or an empty
     * shared dimension scales [c] and reads no operand; a zero [beta] overwrites [c] without reading it.
     *
     * Scaling placement and accumulation order depend on the selected implementation. This includes
     * whether [alpha] scales operand entries or partial sums. Rounding, overflow and non-finite results
     * may therefore differ between implementations and transpose modes.
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
        requireProductOperands(a, false, b, false, "gemm")
        val c = DenseMatrix(a.rows, b.cols)
        gemm(1.0, a, transposeA = false, b, transposeB = false, beta = 0.0, c = c)
        return c
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`).
     * Only the [lower] or upper triangle is written. Built-in implementations snapshot overlap with [c].
     *
     * One product, so [alpha] and the accumulation order carry the latitude [gemm] describes.
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
     *  diagonal included. [workspace] lends the staging an overlap with [y] takes, as in [gemv]. */
    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    public fun symv(
        alpha: Double,
        a: DenseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
        workspace: Workspace? = null,
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

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`). A vector sharing [a]'s buffer is snapshotted, and
     *  [workspace] lends that snapshot. */
    public fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix, workspace: Workspace? = null)

    /**
     * `A(rowOffset + i, columnOffset + j) += alpha · x(xOffset + i) · y(yOffset + j)` for
     * `0 <= i < rows` and `0 <= j < columns`. Entries outside this rectangle are untouched.
     *
     * The destination retains [a]'s column-major leading dimension. Built-in panel kernels update it
     * directly, without copying the rectangle or allocating vector views. All matrix and vector ranges
     * are validated before writing, including when [alpha] is zero or either extent is empty.
     *
     * A vector sharing [a]'s buffer is snapshotted over only its selected range; [workspace] lends that
     * snapshot. A zero [alpha] or an empty rectangle reads no entries and takes no scratch.
     * For [com.eignex.koblas.KoblasEngine.routeOf], use [DenseMatrixOperation.GerWindow] with the rectangle's
     * [rows] and [columns], rather than the full matrix shape.
     */
    @Suppress("LongParameterList") // the rank update and its three independently addressed windows
    public fun ger(
        alpha: Double,
        x: DoubleArray,
        y: DoubleArray,
        a: DenseMatrix,
        rowOffset: Int,
        columnOffset: Int,
        rows: Int,
        columns: Int,
        xOffset: Int = 0,
        yOffset: Int = 0,
        workspace: Workspace? = null,
    )

    /** `A += alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. [x] must be dense or
     *  strided storage, which is what a vendor can address. Overlap staging follows [ger]. */
    @Suppress("LongParameterList") // the BLAS dsyr signature plus the workspace
    public fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean = true, workspace: Workspace? = null)

    /** `A += alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. Operand
     *  storage and overlap staging follow [syr]. */
    @Suppress("LongParameterList") // the BLAS dsyr2 signature plus the workspace
    public fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrix,
        lower: Boolean = true,
        workspace: Workspace? = null,
    )

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Writes only the [lower] or upper triangle and snapshots input overlap in built-in engines.
     *
     * The implementation may accumulate the products separately or in one traversal, with the
     * rounding and overflow differences described by [gemm].
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
     * As in BLAS `dtrsv`, no singularity or conditioning check is performed. Singular inputs have no
     * guaranteed numerical result.
     *
     * The substitution overwrites [x] as it goes, so a triangle sharing that buffer is snapshotted first and
     * [workspace] lends the snapshot.
     */
    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
    public fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        workspace: Workspace? = null,
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

    /** `x = op(T) · x` in place (BLAS `dtrmv`), the product counterpart of [trsv], whose overlap staging and
     *  [workspace] it shares. */
    @Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
    public fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        workspace: Workspace? = null,
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
