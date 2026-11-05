@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass
@file:Suppress("TooManyFunctions", "LongParameterList", "VariableNaming", "FunctionParameterNaming", "MaxLineLength")

package com.eignex.koblas

import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.kernels.unsupported
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.scheduling.sparse.*
import com.eignex.koblas.internal.storage.*

/**
 * Dense and sparse BLAS through one immutable engine.
 *
 * Each BLAS routine is one member. The portable engine runs the shared schedule over its own kernels; a
 * built-in engine such as [koblas] uses one optional host library above its cutoffs and otherwise runs the
 * same schedule over its owned kernels; a vendor engine from [KoblasEngine.vendor] calls its library for the
 * routines it exports and fails for the rest. The companion names exact engines, while [koblas] selects its
 * platform default unless startup configuration overrides it.
 *
 * Level 2 and 3 routines validate shapes, preserve BLAS no-read rules and snapshot an input that shares the
 * destination's storage, borrowing from a [Scratch] when one is given. Level 1 members address primitive array
 * windows: one outside its array throws from the array access, a host library binding checks each window before
 * passing it on, and callers stage destructive overlaps. Rounding, reduction order and unspecified
 * exceptional results belong to the selected arithmetic.
 */
public interface KoblasEngine {

    /** Exact engines for tests and comparisons, also assignable to [koblas] during startup. */
    public companion object {
        /** Opens exactly [vendor] when available. Unsupported operations remain unsupported. */
        @kotlin.jvm.JvmStatic
        public fun vendor(vendor: Vendor): KoblasEngine? = vendorEngine(vendor)

        /** Portable Kotlin across all three levels. */
        @get:kotlin.jvm.JvmStatic
        public val scalar: KoblasEngine get() = ScalarEngine

        /**
         * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
         *
         * On the JVM, [koblas] retains these kernels when available, alongside its selected host library.
         * A window too short or too strided for a vector body runs a portable fallback.
         */
        @get:kotlin.jvm.JvmStatic
        public val simd: KoblasEngine? by lazy { simdBackend() }
    }

    // Engine identity and provenance.

    /** Implementation identifier for diagnostics. */
    public val name: String

    /** Stable provider identifier for diagnostics and internal routing. */
    public val id: String get() = vendor?.vendorName?.lowercase() ?: name

    /** Callbacks supplied by this provider. Shared matrix schedules select arithmetic directly from leaf [providers]. */
    public val implementations: Set<EngineImplementation> get() = emptySet()

    /** Library provenance, when applicable. */
    public val vendor: Vendor? get() = null

    /** Resolved host library, when applicable. */
    public val libraryPath: String get() = ""

    /** Host library version, when applicable. */
    public val version: String get() = ""

    /** Evidence for the host library's compute thread count. */
    public val threadEvidence: ThreadEvidence get() = ThreadEvidence.Unconfirmed

    /** Kernel providers and the selected host library, including capabilities and host provenance. */
    public val providers: List<KoblasEngine> get() = listOf(this)

    // Primitive dense vector windows.

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    public fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Double = unsupported()

    /**
     * Adds `alpha * x[xOff + i]` into `y[yOff + i]` over [len] entries. A zero [alpha] returns without
     * evaluating the products, as required by standalone BLAS AXPY semantics.
     */
    public fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int = 1,
        xStride: Int = 1,
    ): Unit = unsupported()

    /** Scales [len] entries in [v] by [alpha]. */
    public fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit = unsupported()

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    public fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    public fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    public fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = unsupported()

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two primitive vector runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    public fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int = 1,
        yStride: Int = 1,
    ): Unit = unsupported()

    /** Exchanges [len] entries of the two runs. */
    public fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupported()

    /** Copies a primitive vector window. Callers stage destructive overlap before dispatch. */
    public fun copy(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupported()

    /**
     * Returns the plain sum over [len] entries, or zero for an empty run. This is not compensated summation;
     * compiled leaves may reduce vector lanes as a tree.
     */
    public fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    // Indexed sparse vector windows.

    /**
     * `Σ values(k) · dense(indices(k))` over [count] stored entries (Sparse BLAS `usdot`). Repeated and unsorted
     * indices are permitted. Indices are not checked here; [SparseVector] and [SparseMatrix] check theirs once.
     */
    public fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = unsupported()

    /** `Σ x(k) · y(k)` over the positions two stored runs share. Both index runs must be strictly increasing. */
    public fun dot(
        xIndices: IntArray,
        xIndexOffset: Int,
        xValues: DoubleArray,
        xValueOffset: Int,
        xCount: Int,
        yIndices: IntArray,
        yIndexOffset: Int,
        yValues: DoubleArray,
        yValueOffset: Int,
        yCount: Int,
    ): Double = unsupported()

    /**
     * `destination(indices(k)) += alpha · values(k)` (Sparse BLAS `usaxpy`). Indices must be strictly increasing,
     * and [values] must not be [destination]. Every stored product is formed, even for a zero [alpha], which is how a
     * sparse traversal keeps a stored zero's product; the [SparseVector] form skips a zero [alpha] as BLAS does.
     */
    public fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** `destination(indices(k)) = values(k)` (Sparse BLAS `ussc`), with the index and buffer rules of [axpy]. */
    public fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** `values(k) = source(indices(k))` (Sparse BLAS `usga`). [values] must not be [source]. */
    public fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupported()

    /** [gather], then zero each position read in [source] (Sparse BLAS `usgz`). Indices must be strictly increasing. */
    public fun gatherZero(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupported()

    // Dense matrix-vector and rank operations.

    /**
     * `y = alpha · op(A) · x + beta · y` (BLAS `dgemv`), with `op(A)` being `Aᵀ` when [transpose]. `x` is the
     * [xStride]-spaced window of [x] from [xOffset], and `y` likewise; a negative stride walks backwards from its
     * offset, which is the window's first entry.
     *
     * A zero [alpha] or an empty `A` reads neither `A` nor `x`, and a zero [beta] overwrites `y` without reading
     * it. An operand sharing `y`'s storage is snapshotted, from [scratch] when one is given, before the first
     * write.
     */
    public fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val operation = if (transpose) EngineImplementation.GemvTransposed else EngineImplementation.Gemv
        val routed = routedWholeCall(operation, a.rows, a.cols, 1, a !is DenseMatrix)
        if (routed !=
            null
        ) {
            return routed.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)
        }
        prepareMatrixVector(this, alpha, a, transpose, x, xOffset, xStride, beta, y, yOffset, yStride, null, scratch) {
                sa,
                sx,
                so,
                ss,
            ->
            scheduleGemv(alpha, sa, sx, so, ss, beta, y, yOffset, yStride, transpose)
        }
    }

    /**
     * `y = alpha · A · x + beta · y` for a symmetric `A` (BLAS `dsymv`), read from its [lower] or upper triangle,
     * diagonal included. Windows, zero multipliers and overlap follow [gemv].
     */
    public fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Symv, a.rows, a.cols, 1, a !is DenseMatrix)
        if (routed != null) return routed.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)
        requireSquare(a, "symv")
        prepareMatrixVector(this, alpha, a, false, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch) {
                sa,
                sx,
                so,
                ss,
            ->
            scheduleSymv(alpha, sa, sx, so, ss, beta, y, yOffset, yStride, lower)
        }
    }

    /** `A = A + alpha · x · yᵀ` (BLAS `dger`). A vector sharing `A`'s storage is snapshotted. */
    public fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Ger, a.rows, a.cols, 1, a !is DenseMatrix)
        if (routed != null) return routed.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)
        updateRank(alpha, x, xOffset, xStride, y, yOffset, yStride, a, 0, 0, a.rows, a.cols, scratch)
    }

    /**
     * `A(rowOffset + i, columnOffset + j) += alpha · x(xOffset + i) · y(yOffset + j)` for
     * `0 <= i < rows` and `0 <= j < columns`. Entries outside this rectangle are untouched.
     *
     * The destination retains [a]'s column-major leading dimension. Built-in panel kernels update it
     * directly, without copying the rectangle or allocating vector views. All matrix and vector ranges
     * are validated before writing, including when [alpha] is zero or either extent is empty.
     *
     * A vector sharing [a]'s buffer is snapshotted over only its selected range. A zero [alpha] or an empty
     * rectangle reads no entries and takes no scratch.
     */
    @Suppress("LongParameterList") // the rank update and its three independently addressed windows
    public fun ger(
        alpha: Double,
        x: DoubleArray,
        y: DoubleArray,
        a: DenseMatrixRegion,
        rowOffset: Int,
        columnOffset: Int,
        rows: Int,
        columns: Int,
        xOffset: Int = 0,
        yOffset: Int = 0,
        scratch: Scratch? = null,
    ) {
        requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
        updateRank(alpha, x, xOffset, 1, y, yOffset, 1, a, rowOffset, columnOffset, rows, columns, scratch)
    }

    /** `A = A + alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. Overlap follows [ger]. */
    public fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Syr, a.rows, a.rows, 1, a !is DenseMatrix)
        if (routed != null) return routed.syr(alpha, x, xOffset, xStride, a, lower, scratch)
        requireSquare(a, "syr")
        prepareRankUpdate(alpha, x, xOffset, xStride, x, xOffset, xStride, a, scratch) { sx, xo, xs, _, _, _ ->
            scheduleSyr(alpha, sx, xo, xs, a, lower)
        }
    }

    /** `A = A + alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. */
    public fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Syr2, a.rows, a.rows, 1, a !is DenseMatrix)
        if (routed != null) return routed.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)
        requireSquare(a, "syr2")
        prepareRankUpdate(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch) {
                sx,
                xo,
                xs,
                sy,
                yo,
                ys,
            ->
            scheduleSyr2(alpha, sx, xo, xs, sy, yo, ys, a, lower)
        }
    }

    /**
     * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for the [lower] or upper triangle of the square [a], `op`
     * transposing when [transpose] and [unitDiag] taking the diagonal as 1. The window of [x] carries `b` in and
     * `x` out.
     *
     * As in BLAS `dtrsv`, no singularity or conditioning check is performed; singular inputs have no guaranteed
     * numerical result. The substitution overwrites `x` as it goes, so a triangle sharing that storage is
     * snapshotted first.
     */
    public fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val operation = if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv
        val routed = routedWholeCall(operation, a.rows, a.rows, 1, a !is DenseMatrix)
        if (routed != null) return routed.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch) { sa ->
            scheduleTrsv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
        }
    }

    /** `x = op(T) · x` in place (BLAS `dtrmv`), with the flags and overlap of [trsv]. */
    public fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val operation = if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv
        val routed = routedWholeCall(operation, a.rows, a.rows, 1, a !is DenseMatrix)
        if (routed != null) return routed.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch) { sa ->
            scheduleTrmv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
        }
    }

    // Dense matrix products and triangular operations.

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (BLAS `dgemm`), with shapes `op(A): m×k`, `op(B): k×n`, `C: m×n`.
     *
     * An input that shares [c]'s storage is snapshotted first. A zero [alpha] or an empty shared dimension
     * scales [c] and reads no operand; a zero [beta] overwrites [c] without reading it.
     *
     * Scaling placement and accumulation order depend on the selected implementation, including whether
     * [alpha] scales operand entries or partial sums, so rounding, overflow and non-finite results may differ
     * between implementations and transpose modes.
     */
    public fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch? = null,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Gemm, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)
        requireGemmOperands(a, transposeA, b, transposeB, c)
        prepareProduct(alpha, a, b, beta, c, depth, null, null, null, scratch) { sa, sb ->
            scheduleGemm(alpha, sa, transposeA, sb, transposeB, beta, c, depth, scratch)
        }
    }

    /**
     * `C = alpha · op(A) · op(B) + beta · C` in only the selected triangle (Netlib `GEMMTR`, commonly exposed as
     * `gemmt`). `op(A)` is `n×k`, `op(B)` is `k×n`, and [c] is `n×n`. The opposite triangle is neither read nor
     * written. Overlap and multipliers follow [gemm].
     */
    public fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Gemmt, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.gemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch)
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            scheduleGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, lower, scratch)
        }
    }

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right] (BLAS `dsymm`). Only the
     * [lower] or upper triangle of the symmetric [a] is read. Overlap and multipliers follow [gemm].
     */
    public fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Symm, c.rows, c.cols, a.rows, borrowed)
        if (routed != null) return routed.symm(alpha, a, b, beta, c, lower, right, scratch)
        requireSymmOperands(a, b, c, right)
        // One operand serving both sides needs a full snapshot, since the second reads the triangle the first skips.
        val aTriangle = if (sameRegion(a, b)) null else lower
        prepareProduct(alpha, a, b, beta, c, a.rows, null, aTriangle, this, scratch) { sa, sb ->
            scheduleSymm(alpha, sa, sb, beta, c, lower, right, scratch)
        }
    }

    /**
     * `C = alpha · A·Aᵀ + beta · C`, or `alpha · Aᵀ·A + beta · C` when [transpose] (BLAS `dsyrk`). Only the
     * [lower] or upper triangle is written. One product, so [alpha] and the accumulation order carry the
     * latitude [gemm] describes.
     */
    public fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val routed = routedWholeCall(
            EngineImplementation.Syrk,
            c.rows,
            c.cols,
            depth,
            a !is DenseMatrix || c !is DenseMatrix,
        )
        if (routed != null) return routed.syrk(alpha, a, transpose, beta, c, lower, scratch)
        requireSyrkOperands(a, transpose, c)
        prepareProduct(alpha, a, a, beta, c, depth, lower, null, null, scratch) { sa, _ ->
            scheduleSyrk(alpha, sa, transpose, beta, c, lower, scratch)
        }
    }

    /**
     * `C = alpha · (op(A) · op(B)ᵀ + op(B) · op(A)ᵀ) + beta · C` (BLAS `dsyr2k`), where `op` transposes when
     * [transpose]. Only the [lower] or upper triangle is written. The implementation may accumulate the two
     * products separately or in one traversal, with the rounding differences [gemm] describes.
     */
    public fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val depth = if (transpose) a.rows else a.cols
        val borrowed = a !is DenseMatrix || b !is DenseMatrix || c !is DenseMatrix
        val routed = routedWholeCall(EngineImplementation.Syr2k, c.rows, c.cols, depth, borrowed)
        if (routed != null) return routed.syr2k(alpha, a, b, transpose, beta, c, lower, scratch)
        requireSyr2kOperands(a, b, transpose, c)
        prepareProduct(alpha, a, b, beta, c, depth, lower, null, null, scratch) { sa, sb ->
            scheduleSyr2k(alpha, sa, sb, transpose, beta, c, lower, scratch)
        }
    }

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`). Flags follow
     * [trsv]; the right-hand sides are the columns of [b] from the left and its rows from the right.
     */
    public fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(
            EngineImplementation.Trsm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (routed != null) return routed.trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch) { sa ->
            scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, true, scratch)
        }
    }

    /** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`), the counterpart of [trsm]. */
    public fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(
            EngineImplementation.Trmm,
            b.rows,
            b.cols,
            a.rows,
            a !is DenseMatrix || b !is DenseMatrix,
        )
        if (routed != null) return routed.trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch) { sa ->
            scheduleTriangularMatrix(alpha, sa, triangle(lower, unitDiag), transpose, b, right, false, scratch)
        }
    }

    // Sparse matrix operations over validated CSC storage.

    /**
     * Dot product of [column] of [matrix] with [dense], whose length must equal [SparseMatrix.rows].
     * Only stored entries participate, including explicit zeros; absent positions are never evaluated.
     * Reads the current [SparseMatrix.values] on every call and permits that array to be [dense].
     *
     * Built-in implementations reuse the matrix's validated CSC support and call the indexed dot leaf
     * at the column's offsets without copying or rescanning indices. The matrix's structural arrays must
     * remain unchanged, as [SparseMatrix.wrap] requires; this call retains no arrays or column view after returning.
     *
     * @throws IndexOutOfBoundsException if [column] is outside the matrix's columns.
     * @throws IllegalArgumentException if [dense] has the wrong length.
     */
    public fun dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
        requireIndex(column in 0 until matrix.cols) { "column $column is outside [0, ${matrix.cols})" }
        requireShape(matrix.rows == dense.size) { "dotColumn: sizes differ, ${matrix.rows} vs ${dense.size}" }
        val start = matrix.colPointers[column]
        val count = matrix.colPointers[column + 1] - start
        return dot(matrix.rowIndices, start, matrix.values, start, count, dense)
    }

    /**
     * In-place `y = alpha · op(A) · x + beta · y` for a sparse [a], where `op(A)` is `Aᵀ` when [transpose]. A zero
     * [beta] overwrites [y] without reading it, and a zero [alpha] reads neither [a] nor [x].
     *
     * Only stored entries are visited, so an implicit zero of [a] forms no product: a column [a] does not store
     * leaves [y] untouched even where [x] holds an infinity. A stored zero does form its product. [x] and [a]'s
     * coefficients may share [y]'s storage and are snapshotted before [y] is scaled.
     */
    public fun gemv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        // Zero alpha or zero extent reads neither operand; only destination scaling remains.
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return applyBeta(this, y, 0, y.size, beta)
        // Snapshot aliased operands before destination scaling overwrites their coefficients.
        staged(scratch, x, x === y) { stableX ->
            staged(
                scratch,
                a,
                a.values === y,
            ) { stableA -> scheduleSparseGemv(alpha, stableA, stableX, beta, y, transpose) }
        }
    }

    /**
     * `y = alpha · A · x + beta · y` for a square sparse [a] interpreted as symmetric. Exactly the selected
     * [lower] or upper stored triangle is read; a stored off-diagonal entry contributes to both mirrored
     * positions. Multipliers and overlap follow the sparse [gemv].
     */
    public fun symv(
        alpha: Double,
        a: SparseMatrix,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (alpha == 0.0) return applyBeta(this, y, 0, y.size, beta)
        staged(scratch, x, x === y) { stableX ->
            staged(
                scratch,
                a,
                a.values === y,
            ) { stableA -> scheduleSparseSymv(alpha, stableA, stableX, beta, y, lower) }
        }
    }

    /**
     * `C = alpha · A · B + beta · C`, or `C = alpha · B · A + beta · C` when [right], interpreting the square
     * sparse [a] as symmetric from exactly its selected [lower] or upper triangle. A zero [alpha] reads neither
     * input and a zero [beta] does not read [c]. Shared input and destination storage is staged.
     */
    public fun symm(
        alpha: Double,
        a: SparseMatrix,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireSymmOperands(a, b, c, right)
        // Return before staging or loans: an empty axis may be paired with an enormous one.
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
        staged(scratch, a, a.values === c.values) { stableA ->
            stagedRegion(scratch, b, c) { stableB ->
                scaleRegion(c, beta, vectors = this)
                if (right) {
                    multiplySymmetricFromTheRight(this, alpha, stableA, stableB, c, lower)
                } else {
                    multiplySymmetricFromTheLeft(this, alpha, stableA, stableB, c, lower, scratch)
                }
            }
        }
    }

    /**
     * Solve `op(T) · x = b` in place for a sparse triangle, with the flags of the dense [trsv]. A missing diagonal
     * entry represents zero and no singularity check is performed. A right-hand side that is already exactly zero
     * contributes no update, as in reference `dtrsv`.
     */
    public fun trsv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        // Snapshot an aliased triangle before substitution overwrites coefficients needed by later columns.
        staged(scratch, a, a.values === x) { stable ->
            triangularVectorCore(stable, x, lower, transpose, unitDiag, solve = true)
        }
    }

    /** `x = op(T) · x` in place for a sparse triangle. A missing diagonal entry is simply zero. */
    public fun trmv(
        a: SparseMatrix,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        staged(scratch, a, a.values === x) { stable ->
            triangularVectorCore(stable, x, lower, transpose, unitDiag, solve = false)
        }
    }

    /**
     * `C = alpha · op(A) · op(B) + beta · C` (Sparse BLAS `usmm`) for a sparse [a] against a dense [b], or
     * `C = alpha · op(B) · op(A) + beta · C` when [right]. A zero [beta] overwrites [c] without reading it.
     *
     * [right] exists because [a] is distinguished by being the sparse operand and cannot be swapped, so without a
     * side the product of a dense matrix by a sparse one would only be reachable by transposing the result.
     */
    public fun gemm(
        alpha: Double,
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        right: Boolean = false,
        scratch: Scratch? = null,
    ) {
        // Right-side multiplication uses the same derivation with operands reversed.
        if (right) {
            requireGemmOperands(
                b,
                transposeB,
                a,
                transposeA,
                c,
            )
        } else {
            requireGemmOperands(a, transposeA, b, transposeB, c)
        }
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) return scaleRegion(c, beta, vectors = this)
        val depth = if (right) (if (transposeB) b.rows else b.cols) else (if (transposeA) a.rows else a.cols)
        staged(scratch, a, a.values === c.values) { stableA ->
            stagedRegion(scratch, b, c) { stableB ->
                scaleRegion(c, beta, vectors = this)
                if (right) {
                    multiplyFromTheRight(this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, scratch)
                } else {
                    multiplyFromTheLeft(
                        this, alpha, stableA, transposeA, stableB, transposeB, c, c.rows, c.cols, depth, scratch,
                    )
                }
            }
        }
    }

    /**
     * Dense selected-triangle `C = alpha · op(A) · op(A)ᵀ + beta · C` for a sparse [a]. Only the [lower] or upper
     * triangle of [c] is read or written, and products involving implicit zeros are not evaluated.
     */
    public fun syrk(
        alpha: Double,
        a: SparseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val n = if (transpose) a.cols else a.rows
        requireSyrkOperands(a, transpose, c)
        // An empty pattern needs only triangle scaling, avoiding potentially oversized row-indexed adjacency.
        if (alpha == 0.0 || a.nnz == 0 || n == 0) return scaleRegion(c, beta, lower)
        staged(scratch, a, a.values === c.values) { stableA ->
            scaleRegion(c, beta, lower)
            withSymmetricRankScratch(scratch, n, stableA.rows, stableA.nnz) {
                    sums,
                    touchedAt,
                    touched,
                    rowPointers,
                    adjacentColumns,
                    adjacentPositions,
                    rowCursor,
                ->
                symmetricRankInto(
                    alpha, stableA, transpose, c, lower, sums, touchedAt, touched,
                    rowPointers, adjacentColumns, adjacentPositions, rowCursor,
                )
            }
        }
    }

    /**
     * `B = alpha · op(T)⁻¹ · B` in place, or `B = alpha · B · op(T)⁻¹` when [right] (Sparse BLAS `ussm`), with the
     * flags of the sparse [trsv]. A missing diagonal entry behaves as zero and singularity is not reported. From
     * the right a stored zero off the diagonal is skipped, as reference `dtrsm` skips a zero coefficient.
     */
    public fun trsm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, scratch)

    /**
     * `B = alpha · op(T) · B` in place, or `B = alpha · B · op(T)` when [right]. Zero or missing diagonal entries
     * participate normally. From the right a stored zero off the diagonal is skipped, as reference `dtrmm` skips
     * a zero coefficient.
     */
    public fun trmm(
        a: SparseMatrix,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        right: Boolean = false,
        alpha: Double = 1.0,
        scratch: Scratch? = null,
    ): Unit = sparseTriangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, scratch)
}

/**
 * The shared BLAS engine used by convenience operations and libraries using this Koblas instance.
 *
 * Initialized from `koblas.engine` on JVM, then `KOBLAS_ENGINE`, or automatic platform selection.
 * Applications may assign an engine during startup, before libraries use Koblas or concurrent work begins.
 * Callers own startup order and safe publication; reads and writes are unsynchronized.
 *
 * The initial default is constructed before an assignment, so launch settings must be valid even when
 * overridden programmatically. Assigned engines are used as supplied, including any unsupported operations.
 */
@get:kotlin.jvm.JvmName("getDefault")
@set:kotlin.jvm.JvmName("setDefault")
public var koblas: KoblasEngine = engineFromSetting(platformEngineSetting())
