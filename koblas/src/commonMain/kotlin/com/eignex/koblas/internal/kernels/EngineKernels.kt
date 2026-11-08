@file:Suppress("TooManyFunctions", "LongParameterList", "MaxLineLength") // the kernel seam shared schedules call into

package com.eignex.koblas.internal.kernels

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.engine.implementationFor
import com.eignex.koblas.internal.operations.*
import com.eignex.koblas.internal.scheduling.dense.*
import com.eignex.koblas.internal.scheduling.sparse.*
import com.eignex.koblas.internal.storage.*

/**
 * The kernels a shared schedule calls into, and the hints routing reads before it calls them.
 *
 * Portable, Vector API and vendor engines implement it; the BLAS routines on [KoblasEngine] are what callers
 * see. A kernel an engine does not supply fails. Built-in engines bind fixed kernel roles and provide
 * scalar coverage for omitted operations.
 */
internal interface EngineKernels : KoblasEngine {

    // Primitive dense vector windows.

    /** Returns `sum(a[aOff + i] * b[bOff + i])` over [len] entries, or zero for an empty run. */
    fun dot(
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
    fun axpy(
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
    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int = 1): Unit = unsupported()

    /**
     * Returns the rescaled Euclidean norm over [len] entries, or zero for an empty run. Implementations
     * must avoid the avoidable overflow and underflow of a plain square-sum followed by a square root.
     */
    fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /** Sum of absolute values over [len] entries, or zero for an empty run. */
    fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    /**
     * Zero-based index within the selected run of a maximum absolute value, or `-1` when empty. A nonempty
     * run of zeros returns `0`.
     *
     * Which index is returned when the maximum is not unique, and what happens when the largest magnitude is
     * a NaN, is this implementation's. `idamax` specifies neither: the portable kernels compare strictly, so
     * ties go to the first and a NaN loses to a later finite entry, while a vendor may report the NaN's index
     * and may resolve a tie the other way for a negatively spaced run.
     */
    fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Int = unsupported()

    /**
     * Applies the plane rotation `(c, s, -s, c)` to two primitive vector runs. Each pair is loaded before either
     * result is stored. Equal runs are therefore safe, but callers must snapshot any other overlap whose
     * stores could precede a later input load.
     */
    @Suppress("LongParameterList")
    fun rot(
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
    fun swap(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int = 1,
        bStride: Int = 1,
    ): Unit = unsupported()

    /** Copies a primitive vector window. Callers stage destructive overlap before dispatch. */
    fun copy(
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
    fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int = 1): Double = unsupported()

    // Indexed sparse vector windows.

    /**
     * `Σ values(k) · dense(indices(k))` over [count] stored entries (Sparse BLAS `usdot`). Repeated and unsorted
     * indices are permitted. Indices are not checked here; [SparseVector] and [SparseMatrix] check theirs once.
     */
    fun dot(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        dense: DoubleArray,
    ): Double = unsupported()

    /** `Σ x(k) · y(k)` over the positions two stored runs share. Both index runs must be strictly increasing. */
    fun dot(
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
    fun axpy(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        alpha: Double,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** `destination(indices(k)) = values(k)` (Sparse BLAS `ussc`), with the index and buffer rules of [axpy]. */
    fun scatter(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        destination: DoubleArray,
    ): Unit = unsupported()

    /** `values(k) = source(indices(k))` (Sparse BLAS `usga`). [values] must not be [source]. */
    fun gather(
        indices: IntArray,
        indexOffset: Int,
        values: DoubleArray,
        valueOffset: Int,
        count: Int,
        source: DoubleArray,
    ): Unit = unsupported()

    /** [gather], then zero each position read in [source] (Sparse BLAS `usgz`). Indices must be strictly increasing. */
    fun gatherZero(
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
    fun gemv(
        alpha: Double,
        a: DenseMatrix,
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
        val routed = routedWholeCall(operation, a.rows, a.cols, 1, !a.isWholeBuffer)
        if (routed !=
            null
        ) {
            return routed.kernels.gemv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch)
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
    fun symv(
        alpha: Double,
        a: DenseMatrix,
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
        val routed = routedWholeCall(EngineImplementation.Symv, a.rows, a.cols, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.symv(alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch)
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
    fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrix,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Ger, a.rows, a.cols, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.ger(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch)
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
    fun ger(
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
        scratch: Scratch? = null,
    ) {
        requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
        updateRank(alpha, x, xOffset, 1, y, yOffset, 1, a, rowOffset, columnOffset, rows, columns, scratch)
    }

    /** `A = A + alpha · x · xᵀ` (BLAS `dsyr`), writing only the [lower] or upper triangle. Overlap follows [ger]. */
    fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrix,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Syr, a.rows, a.rows, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.syr(alpha, x, xOffset, xStride, a, lower, scratch)
        requireSquare(a, "syr")
        prepareRankUpdate(alpha, x, xOffset, xStride, x, xOffset, xStride, a, scratch) { sx, xo, xs, _, _, _ ->
            scheduleSyr(alpha, sx, xo, xs, a, lower)
        }
    }

    /** `A = A + alpha · (x · yᵀ + y · xᵀ)` (BLAS `dsyr2`), writing only the [lower] or upper triangle. */
    fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrix,
        lower: Boolean = true,
        scratch: Scratch? = null,
    ) {
        val routed = routedWholeCall(EngineImplementation.Syr2, a.rows, a.rows, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.syr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, lower, scratch)
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
    fun trsv(
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val operation = if (transpose) EngineImplementation.TrsvTransposed else EngineImplementation.Trsv
        val routed = routedWholeCall(operation, a.rows, a.rows, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.trsv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch) { sa ->
            scheduleTrsv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
        }
    }

    /** `x = op(T) · x` in place (BLAS `dtrmv`), with the flags and overlap of [trsv]. */
    fun trmv(
        a: DenseMatrix,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiag: Boolean = false,
        scratch: Scratch? = null,
    ) {
        val operation = if (transpose) EngineImplementation.TrmvTransposed else EngineImplementation.Trmv
        val routed = routedWholeCall(operation, a.rows, a.rows, 1, !a.isWholeBuffer)
        if (routed != null) return routed.kernels.trmv(a, x, xOffset, xStride, lower, transpose, unitDiag, scratch)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch) { sa ->
            scheduleTrmv(sa, x, xOffset, xStride, lower, transpose, unitDiag)
        }
    }

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
    fun dotColumn(matrix: SparseMatrix, column: Int, dense: DoubleArray): Double {
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
    fun gemv(
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
    fun symv(
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
     * Solve `op(T) · x = b` in place for a sparse triangle, with the flags of the dense [trsv]. A missing diagonal
     * entry represents zero and no singularity check is performed. A right-hand side that is already exactly zero
     * contributes no update, as in reference `dtrsv`.
     */
    fun trsv(
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
    fun trmv(
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

    /** Whether attribution can depend on the values processed by this contribution. */
    fun dynamicAttribution(implementation: EngineImplementation): Boolean = false

    /** Whether this contribution requires unit strides. */
    fun requiresUnitStride(implementation: EngineImplementation): Boolean = false

    /** Smallest run the contribution can execute. The built-in engine checks it before calling the provider. */
    fun minimumLength(implementation: EngineImplementation): Int = 0

    /** The body a panel of these extents reaches, which is this engine's own unless it hands some shapes on. */
    fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true): String = name

    /** The bodies a [productBlock] of these extents reaches, in the order its row sweep takes them. */
    fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
        if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(name)

    /** The bodies a diagonal block of [order] steps over [sides] right-hand sides reaches. */
    fun implementationsFor(order: Int, sides: Int, contiguous: Boolean): List<String> =
        if (order <= 0 || sides <= 0) emptyList() else listOf(name)

    /**
     * How many columns of [work] this engine recommends processing together, for a panel of [rows] by
     * [columns].
     *
     * Always at least one and never more than [columns] when [columns] is positive. The answer is a local
     * tuning choice made from the resolved vector width and small measured rules, and it is not a lane count:
     * a group of four columns and four lanes are different numbers that happen to agree on some machines.
     *
     * [contiguous] is part of the question for the same reason it is part of [implementationFor]'s: an engine
     * whose body over adjacent entries is a different body may want a different amount of work in it. A
     * caller that will hand over a strided window has to say so, or it is given a recommendation for a body
     * it is not going to reach. A scheduling of its own may then take less than what was recommended, for a
     * cache or a workspace it is keeping within; passing less is always correct.
     */
    fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean = true): Int = unsupported()

    /**
     * Whether a panel of these extents is worth copying into adjacent rows before it is handed over.
     *
     * A caller that can copy a strided window into adjacent storage needs to know whether the copy buys
     * anything before it pays for it, and the answer is the engine's, because only the engine knows what
     * its body does with the two layouts. A portable body reads one entry at a time either way and says no
     * at every extent. A vector body says yes where the panel is wide enough that the arithmetic it gains
     * outweighs a pass over the data, which is not the same width as the one where its body starts running:
     * one lane block is a single vector operation per column, and a copy is not repaid by that.
     *
     * This is a recommendation about copying rather than a claim about which body runs, and the two are
     * asked separately. Where it is true, [implementationFor] also names a different body for the two
     * layouts; where it is false the body may still differ, and the caller has simply been told not to pay
     * for the difference.
     */
    fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean = unsupported()

    /**
     * `y(c) = alpha · Σ a(i, c) · x(i) + beta · y(c)` for each of [columns] columns.
     *
     * The destination is [columns] entries from [yOffset], spaced [yStride] apart. A zero [beta] overwrites
     * without reading what is there, as every BLAS routine carrying one does. With no rows every sum is
     * empty and so zero, and each output is still written from that and whatever [beta] contributes; with no
     * columns nothing happens at all.
     */
    fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = unsupported()

    /**
     * `y(i) += Σ (alpha · x(c)) · a(i, c)` over [rows] destination entries and [columns] panel columns.
     *
     * Scaling and reduction grouping may vary between implementations; grouping does not promise identical
     * rounding or exceptional results.
     */
    fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = unsupported()

    /**
     * One pass over a panel that both updates a destination window and reduces the panel against a vector.
     *
     * `y(i) += Σ (alpha · coefficients(c)) · a(i, c)` over the [rows] entries of the window, and
     * `sums(c) += alpha · Σ a(i, c) · x(i)` for each column. This is a symmetric traversal's second half
     * removed: the stored triangle is read once and serves both the column below the diagonal and the row
     * to its left.
     *
     * [sums] is the caller's, which is where several results go rather than into a returned object. Its
     * window must not overlap [y]'s, which for a symmetric traversal it never does: the reduction lands on
     * the diagonal side of the rows the update wrote. Nor may [x]'s window overlap [y]'s, because an
     * implementation is free to read each `x(i)` before or after it writes `y(i)`.
     */
    fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
        xStride: Int = 1,
        yStride: Int = 1,
        coefficientStride: Int = 1,
        sumStride: Int = 1,
    ): Unit = unsupported()

    /**
     * `a(i, c) += (alpha · coefficients(c)) · x(i)`, the rank-one update of a panel of destination columns.
     *
     * The destination is the panel here and [x] is the source window, which is the opposite direction from
     * [columnUpdate] and the reason it is its own entry rather than that one with the arguments swapped.
     */
    fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ): Unit = unsupported()

    /**
     * `a(i, c) += alpha · (x(i) · u(c) + y(i) · v(c))`, the symmetric rank-two update of a panel.
     *
     * Two [rankUpdate] calls compute the same thing, and that is what this does unless an engine overrides it.
     * One pass loads and stores each destination entry once instead of twice, which is the limit a narrow
     * vector unit reaches first.
     */
    fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ): Unit = unsupported()

    /**
     * `y(i) += Σ (alpha · values(c)) · a(i, indices(c))` over a panel whose columns an index array selects.
     *
     * The sparse counterpart of [columnUpdate], and the shape a sparse column takes when its dense operand
     * carries several right-hand sides. [rows] is the group of right-hand sides, which is the axis with no
     * dependences between its entries and therefore the one an engine may vectorise; [columns] is how many
     * stored entries the column contributes, read as `indices` and `values` from [fromIndex] onward.
     *
     * An entry of the panel is at `aOffset + i · rowStride + indices(c) · indexStride`, which is two strides
     * rather than a leading dimension because either axis may be the contiguous one. A dense block in
     * column-major storage has its right-hand sides a leading dimension apart, and one staged into
     * right-hand-side-major order has them adjacent; the arithmetic is the same and only the strides differ.
     * [implementationFor] answers which body a given panel reaches, and a caller that wants the vector one
     * has to make [rowStride] one for it.
     *
     * The destination is [rows] adjacent entries from [yOffset] and is accumulated into. Each indexed stored
     * position participates, including a stored zero; absent positions are never evaluated. This sparse
     * structural rule also applies to [indexedRankUpdate] and [indexedCoupledUpdate], independently of the
     * zero-coefficient latitude of dense updates.
     */
    fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ): Unit = unsupported()

    /**
     * `a(i, indices(c)) += (alpha · values(c)) · x(i)`, the indexed counterpart of [rankUpdate].
     *
     * The destination is the panel and [x] is the source window of [rows] adjacent entries, addressed as
     * [indexedColumnUpdate] describes.
     *
     * Every logical entry this call writes must have an address of its own. Two things are needed for that
     * and the caller owns both: the selected positions must be distinct, which a validated CSC column's
     * ascending row indices are, and the two strides must describe a panel that does not fold onto itself,
     * which they fail to do where a stride is zero or where one is a multiple of the other inside the
     * extents in play. An engine may hold several entries in registers at once, so two logical entries
     * sharing an address would lose an update rather than apply both. This is the one precondition these two
     * leaves do not share: a reduction may read a position twice, a scatter may not write one twice.
     *
     * These internal primitive windows are trusted; validation belongs to the common matrix entry points. A public matrix call passes windows of its
     * own storage, where the strides come from a layout that already holds.
     */
    fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ): Unit = unsupported()

    /**
     * One pass over an indexed panel that both scatters into the positions it selects and reduces them.
     *
     * For every position `c` of the run, with `t = alpha · values(c)`:
     * `a(i, indices(c)) += t · b[pivot + i · rowStride]`, and `sums(i) += t · b(i, indices(c))` unless `c`
     * is [excluded].
     *
     * This is [indexedRankUpdate] and [indexedColumnUpdate] over one walk of the same run, and the indexed
     * counterpart of [coupledUpdateDot]. It is here for the same reason that one is: a symmetric traversal
     * stores one triangle, every stored entry serves the column it sits in and the row it mirrors into, and
     * reading it once is half the index work of reading it twice. One multiplier serves both halves, so a
     * caller applies its own scaling through [alpha] rather than pre-scaling one side of the pair.
     *
     * Both halves read [b]: the scattered one reads the [rows] entries from [pivot] spaced [rowStride]
     * apart, which is one indexed column of the same window, and the reduced one reads the column each
     * position selects. A caller therefore hands over no coefficients of its own, because gathering that
     * column into a scratch first would be a pass over it per column of the sparse operand.
     *
     * [a] and [b] are windows of the same shape addressed alike, from [offset] with the two strides
     * [indexedColumnUpdate] describes; a symmetric product passes its destination and its source, which are
     * different arrays. The reduction lands in [rows] entries from [sumOffset] spaced [rowStride] apart.
     * This window may coincide with one indexed column of [a], with each reduction entry overlapping only
     * the scatter entry for the same logical right-hand side; an overlap between different right-hand sides
     * is unsupported. Where both updates target the same entry, the scatter is applied before the reduction
     * for each stored position. All source windows remain disjoint from both destination windows.
     *
     * [excluded] is a position relative to this run, in `0 until columns`, whose product is scattered but
     * not reduced; `-1` excludes none. The column of [b] it selects is not read, though [pivot] still is,
     * because the scatter runs at that position like any other. A symmetric column excludes its own diagonal
     * because that contribution is already scattered into the pivot row.
     *
     * Stored positions participate even when their values are zero. The reduction accumulates into its
     * destination; term grouping and rounding may vary subject to the overlap rule above.
     */
    @Suppress("LongParameterList") // the panel, its two windows, the run, and both results
    fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
        destinationOffset: Int = offset,
        destinationRowStride: Int = rowStride,
        destinationIndexStride: Int = indexStride,
    ): Unit = unsupported()

    /**
     * Destination rows one register tile accumulates, which is the group a left panel is packed in.
     *
     * Positive and fixed for the life of an implementation.
     */
    val tileRows: Int get() = unsupported()

    /** Destination columns one register tile accumulates, which is the group a right panel is packed in. */
    val tileColumns: Int get() = unsupported()

    /**
     * Whether a product of these logical extents is worth packing into this engine's tiles at all.
     *
     * Packing costs a pass over both operands and buys contiguous, tile-shaped reads for the whole product.
     * The arithmetic grows with `rows · columns · depth` and the copy with `rows · depth + depth · columns`,
     * so a product with any small extent pays for a copy it cannot amortise, and a matrix-vector shaped call
     * pays for one it can never amortise at all. Where this answers false the caller runs the product as
     * panel work over the operands where they are, which is what avoids copying both of them into tiles. It
     * is not a promise that nothing is copied: that path may still gather one strided coefficient column,
     * borrow a destination column to accumulate into, and stage an operand that shares the destination.
     *
     * The crossover is local: a tile that finishes more arithmetic per loaded value pays off a copy sooner.
     */
    fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean = unsupported()

    /**
     * `C = alpha · A · B + beta · C` over one `rows` by `columns` window, accumulating `depth` steps.
     *
     * [packedA] holds the left operand in [tileRows]-wide groups and [packedB] the right in
     * [tileColumns]-wide ones. Within a group, depth steps are adjacent blocks of that width, so entry
     * `(i, p)` of the left operand sits at `aOffset + (i / tileRows) · aGroupStride + p · tileRows +
     * i % tileRows`, and entry `(p, j)` of the right at `bOffset + (j / tileColumns) · bGroupStride +
     * p · tileColumns + j % tileColumns`.
     *
     * The group strides are `depth · tileRows` and `depth · tileColumns` for transient panels.
     *
     * The destination window starts at [cOffset] with columns [ldc] apart. With no rows or no columns
     * nothing happens and nothing is read. With no depth the product is empty, so the window becomes
     * `beta · C` and neither packed panel is read.
     */
    fun productBlock(
        alpha: Double,
        packedA: DoubleArray,
        aOffset: Int,
        aGroupStride: Int,
        packedB: DoubleArray,
        bOffset: Int,
        bGroupStride: Int,
        rows: Int,
        columns: Int,
        depth: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        ldc: Int,
    ): Unit = unsupported()

    /**
     * How many right-hand sides this engine recommends substituting at once, for a block of [order] steps
     * over [sides] of them.
     *
     * Always at least one and never more than [sides] when that is positive. It is a grouping and not a lane
     * count, and it is independent of the register tile, the panel grouping and the diagonal block the
     * scheduling chose: a diagonal block is as wide as the dependency chain is long, and this is how much of
     * the work beside it is done together.
     */
    fun rightHandSideGroup(order: Int, sides: Int): Int = unsupported()

    /**
     * Whether a group of right-hand sides this far apart is worth copying into adjacent storage first.
     *
     * The copy is two passes over a block of the caller's data, out and back, and it buys a substitution
     * whose independent values lie next to each other. An engine whose arithmetic is a value at a time gains
     * nothing from that and answers false, and so does one whose right-hand sides are already adjacent.
     * Where this answers true the caller gathers.
     */
    fun gathersRightHandSides(rhsStride: Int, order: Int, sides: Int): Boolean = unsupported()

    /**
     * Solves `M · X = X` in place over one diagonal block, for every right-hand side at once.
     *
     * Substitutes forward when [lower] and backward when it is not, dividing each finished step by its
     * diagonal coefficient unless [unitDiag]. No singularity check or particular result for singular or
     * non-finite inputs is promised.
     */
    fun diagonalSolve(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int = 0,
    ): Unit = unsupported()

    /**
     * Multiplies `X = M · X` in place over one diagonal block, for every right-hand side at once.
     *
     * The order is the one that leaves no step needed after it has been overwritten, which runs opposite to
     * the solve's: a lower block is substituted from its last step upward, because a step is the sum of the
     * steps at or before it and those must still hold what they arrived with.
     */
    fun diagonalMultiply(
        t: DoubleArray,
        n: Int,
        start: Int,
        size: Int,
        transposed: Boolean,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
        xOffset: Int,
        orderStride: Int,
        rhsStride: Int,
        sides: Int,
        tOffset: Int = 0,
    ): Unit = unsupported()
}

internal fun KoblasEngine.unsupported(): Nothing =
    throw UnsupportedOperationException("This operation is not supplied by $name")

/** Access to the primitive seam is confined to storage adapters and shared schedules. */
internal val KoblasEngine.kernels: EngineKernels get() = this as? EngineKernels ?: unsupported()
