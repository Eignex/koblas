@file:Suppress("FunctionParameterNaming", "VariableNaming") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.Scratch
import com.eignex.koblas.borrow
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.dense.scalarEvaluatedAxpy
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.operations.staged
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional
import kotlin.jvm.JvmInline
import kotlin.math.min

/** Schedule-owned staging capacity, independent of the backend's recommended panel width. */
private const val CONTIGUOUS_RHS_PANEL: Int = 64

/**
 * Minimum stored entries traversed per dense element copied. Local measurements favored staging from
 * ratio four; sparser products keep the caller's layout. Gather copies the source once, scatter copies
 * the destination in and out, so the same ratio admits gather at half the density.
 */
private const val STAGED_RHS_CROSSOVER: Int = 4

/**
 * Choose backend grouping, capped by [CONTIGUOUS_RHS_PANEL], and stage only when adjacent arithmetic repays
 * its copy. Already-adjacent inputs and portable kernels need no staging. [RhsPlan] encodes staging in the
 * width's sign without a plan allocation. [rows] selects grouping; [reduction] selects gather or scatter.
 * [copiedPerSide] counts source copies once and destination copies twice; `Long` avoids cost overflow.
 */
@Suppress("LongParameterList") // the seam, the two extents, the copy's cost and what the direction admits
internal fun planRightHandSides(
    kernels: KernelProvider,
    rows: Int,
    sides: Int,
    entries: Int,
    copiedPerSide: Long,
    nativelyContiguous: Boolean,
    reduction: Boolean = false,
): RhsPlan {
    if (sides <= 0) return RhsPlan(0)
    val adjacent = min(
        sides,
        min(
            kernels.sparseRightHandSideGroup(rows, sides, contiguous = true, reduction = reduction),
            CONTIGUOUS_RHS_PANEL,
        ),
    )
    if (nativelyContiguous) return RhsPlan(adjacent)
    val strided = kernels.sparseRightHandSideGroup(rows, sides, contiguous = false, reduction = reduction)
    // Packing must beat the strided traversal and fit the staged panel in one array.
    val staged = entries > 0 &&
        kernels.sparsePrefersAdjacentSides(adjacent, entries, reduction) &&
        entries.toLong() >= STAGED_RHS_CROSSOVER.toLong() * copiedPerSide &&
        adjacent * copiedPerSide <= Int.MAX_VALUE
    return RhsPlan(if (staged) -adjacent else strided)
}

/**
 * Gather products stage the dense source once; scatter products stage the destination in and out.
 * Already-adjacent right-hand sides need no copy.
 */
internal fun productRhsPlan(
    kernels: KernelProvider,
    rows: Int,
    sides: Int,
    depth: Int,
    entries: Int,
    transposeSparse: Boolean,
    transposeDense: Boolean,
): RhsPlan = planRightHandSides(
    kernels,
    rows,
    sides,
    entries,
    copiedPerSide = if (transposeSparse) depth.toLong() else 2L * rows,
    nativelyContiguous = transposeSparse && transposeDense,
    reduction = transposeSparse,
)

/**
 * The plan a symmetric sparse product takes, whose staging copies both dense blocks and writes one back.
 *
 * Three passes over a right-hand side against the two a general product's destination costs, which is why
 * the two weigh the copy differently.
 */
internal fun symmetricRhsPlan(kernels: KernelProvider, order: Int, sides: Int, entries: Int): RhsPlan =
    planRightHandSides(kernels, order, sides, entries, copiedPerSide = 3L * order, nativelyContiguous = false)

/** Right-hand-side grouping, with a negative encoding when the dense block needs staging. */
@JvmInline
internal value class RhsPlan(private val encoded: Int) {
    val width: Int get() = if (encoded < 0) -encoded else encoded
    val staged: Boolean get() = encoded < 0
}

/**
 * Copy [actual] right-hand sides between the dense block and [panel], with [width] adjacent sides per index.
 * Dense address: `origin + r · rhsStride + i · indexStride`, for `i` below [count].
 * Copyback includes untouched positions, preserving their bits and negative zeros.
 */
@Suppress("LongParameterList") // the block window with its two strides, and the staged panel
internal fun copyRhsPanel(
    values: DoubleArray,
    origin: Int,
    rhsStride: Int,
    indexStride: Int,
    count: Int,
    actual: Int,
    panel: DoubleArray,
    width: Int,
    toPanel: Boolean,
) {
    for (index in 0 until count) {
        val at = origin + index * indexStride
        val staged = index * width
        for (rhs in 0 until actual) {
            if (toPanel) {
                panel[staged + rhs] = values[at + rhs * rhsStride]
            } else {
                values[at + rhs * rhsStride] = panel[staged + rhs]
            }
        }
    }
}

/**
 * Runs [body] over [actual] right-hand sides of a dense block, staged by [copyRhsPanel] into [panel] when
 * [staged] and in place otherwise. [body] receives the window to address as the block is addressed, and
 * [writeBack] copies a staged panel back afterwards.
 */
@Suppress("LongParameterList") // the block window with its two strides, and the staged panel
internal inline fun withRhsPanel(
    values: DoubleArray,
    origin: Int,
    rhsStride: Int,
    indexStride: Int,
    count: Int,
    actual: Int,
    staged: Boolean,
    panel: DoubleArray,
    width: Int,
    writeBack: Boolean,
    body: (dense: DoubleArray, offset: Int, rhsStride: Int, indexStride: Int) -> Unit,
) {
    if (staged) copyRhsPanel(values, origin, rhsStride, indexStride, count, actual, panel, width, toPanel = true)
    body(
        if (staged) panel else values,
        if (staged) 0 else origin,
        if (staged) 1 else rhsStride,
        if (staged) width else indexStride,
    )
    if (staged && writeBack) {
        copyRhsPanel(values, origin, rhsStride, indexStride, count, actual, panel, width, toPanel = false)
    }
}

/**
 * Sparse-left product, reusing each CSC walk over a right-hand-side panel. Transposed products reduce into
 * adjacent accumulators; untransposed products scatter into the dense destination, whose layout determines
 * whether adjacent staging enables vector arithmetic.
 */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    transposeA: Boolean,
    bValues: DoubleArray,
    bOffset: Int,
    bLeadingDimension: Int,
    transposeB: Boolean,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    m: Int,
    n: Int,
    k: Int,
    arrays: Scratch?,
) {
    // Return before staging or traversing: an empty axis may be paired with an enormous one.
    if (m == 0 || n == 0 || k == 0) return
    val ldb = bLeadingDimension
    val bRhsStride = if (transposeB) 1 else ldb
    val bIndexStride = if (transposeB) ldb else 1
    val plan = productRhsPlan(kernels, m, n, k, aValues.size, transposeA, transposeB)
    val width = plan.width
    // Single-right-hand-side groups reach no panel; avoid retaining unused scratch in the arrays.
    arrays.borrowOptional(if (width > 1 && n > 1) width else 0) { work ->
        arrays.borrowOptional(if (plan.staged) width * (if (transposeA) k else m) else 0) { panel ->
            forEachPanel(n, width) { columnStart, actual ->
                if (transposeA) {
                    gatherPanel(
                        kernels,
                        alpha,
                        aValues,
                        aPointers,
                        aIndices,
                        aColumns,
                        bValues,
                        bOffset,
                        cValues,
                        cOffset,
                        cLeadingDimension,
                        k,
                        columnStart,
                        actual,
                        width,
                        plan.staged,
                        bRhsStride,
                        bIndexStride,
                        panel,
                        work,
                    )
                } else {
                    scatterPanel(
                        kernels,
                        alpha,
                        aValues,
                        aPointers,
                        aIndices,
                        aColumns,
                        bValues,
                        bOffset,
                        cValues,
                        cOffset,
                        cLeadingDimension,
                        m,
                        columnStart,
                        actual,
                        width,
                        plan.staged,
                        bRhsStride,
                        bIndexStride,
                        panel,
                        work,
                    )
                }
            }
        }
    }
}

/** One panel of right-hand sides of `C += alpha · Aᵀ · op(B)`, staged or in the caller's own layout. */
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun gatherPanel(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    bValues: DoubleArray,
    bOffset: Int,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    k: Int,
    columnStart: Int,
    actual: Int,
    width: Int,
    staged: Boolean,
    bRhsStride: Int,
    bIndexStride: Int,
    panel: DoubleArray,
    work: DoubleArray,
) {
    val origin = bOffset + columnStart * bRhsStride
    val cOrigin = cOffset + columnStart * cLeadingDimension
    withRhsPanel(
        bValues, origin, bRhsStride, bIndexStride, k, actual, staged, panel, width, writeBack = false,
    ) { source, offset, rhsStride, indexStride ->
        if (actual == 1) {
            forEachSparseColumn(aPointers, aColumns) { outputRow, from, to ->
                sparseGatherProductColumn(
                    alpha, aIndices, aValues, from, to, source, offset, indexStride, cValues, cOrigin + outputRow,
                )
            }
        } else {
            forEachSparseColumn(aPointers, aColumns) { outputRow, from, to ->
                kernels.sparseGatherProductPanel(
                    alpha, aIndices, aValues, from, to,
                    source, offset, rhsStride, indexStride, actual,
                    cValues, cOrigin + outputRow, cLeadingDimension, work,
                )
            }
        }
    }
}

/** One panel of right-hand sides of `C += alpha · A · op(B)`, with the destination staged or in place. */
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun scatterPanel(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    bValues: DoubleArray,
    bOffset: Int,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    m: Int,
    columnStart: Int,
    actual: Int,
    width: Int,
    staged: Boolean,
    bRhsStride: Int,
    bIndexStride: Int,
    panel: DoubleArray,
    work: DoubleArray,
) {
    val origin = cOffset + columnStart * cLeadingDimension
    val bOrigin = bOffset + columnStart * bRhsStride
    withRhsPanel(
        cValues, origin, cLeadingDimension, 1, m, actual, staged, panel, width, writeBack = true,
    ) { destination, offset, rhsStride, indexStride ->
        if (actual == 1) {
            forEachSparseColumn(aPointers, aColumns) { inner, from, to ->
                sparseScatterProductColumn(
                    alpha, aIndices, aValues, from, to,
                    bValues, bOrigin + inner * bIndexStride, destination, offset, indexStride,
                )
            }
        } else {
            forEachSparseColumn(aPointers, aColumns) { inner, from, to ->
                kernels.sparseScatterProductPanel(
                    alpha, aIndices, aValues, from, to,
                    bValues, bOrigin + inner * bIndexStride, bRhsStride, actual,
                    destination, offset, rhsStride, indexStride, work,
                )
            }
        }
    }
}

/**
 * Symmetric sparse-left product. One panel geometry serves both dense operands; staging copies both in
 * and only the destination out, costing three passes per side versus a general scatter product's two.
 */
@Suppress("LongParameterList") // the operands, the triangle flag, and the scratch
internal fun multiplySymmetricFromTheLeft(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aRows: Int,
    aColumns: Int,
    bValues: DoubleArray,
    bOffset: Int,
    bColumns: Int,
    bLeadingDimension: Int,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    lower: Boolean,
    arrays: Scratch?,
) {
    val n = aRows
    val sides = bColumns
    val ld = bLeadingDimension
    val ldc = cLeadingDimension
    if (n == 0 || sides == 0) return
    val plan = symmetricRhsPlan(kernels, n, sides, aValues.size)
    val width = plan.width
    val panelSize = if (plan.staged) width * n else 0
    arrays.borrowOptional(panelSize) { source ->
        arrays.borrowOptional(panelSize) { destination ->
            forEachPanel(sides, width) { columnStart, actual ->
                val bOrigin = bOffset + columnStart * ld
                val cOrigin = cOffset + columnStart * ldc
                withRhsPanel(
                    bValues, bOrigin, ld, 1, n, actual, plan.staged, source, width, writeBack = false,
                ) { sourceValues, offset, rhsStride, indexStride ->
                    withRhsPanel(
                        cValues, cOrigin, ldc, 1, n, actual, plan.staged, destination, width, writeBack = true,
                    ) { into, destinationOffset, destinationStride, _ ->
                        forEachSparseColumn(aPointers, aColumns) { column, from, to ->
                            if (actual == 1) {
                                sparseSymmetricLeftColumnSingle(
                                    alpha, column, aIndices, aValues, from, to,
                                    sourceValues, into, offset, indexStride, lower, destinationOffset,
                                )
                            } else {
                                kernels.sparseSymmetricLeftColumn(
                                    alpha, column, aIndices, aValues, from, to,
                                    sourceValues, into, offset, rhsStride, indexStride, actual, lower,
                                    destinationOffset, destinationStride,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * `C += alpha · op(B) · op(A)` over the sparse operand's CSC columns.
 *
 * With the sparse operand on the right every update is a whole column of the dense block, so a transposed
 * dense operand is staged once into column-major order rather than read across its rows once per entry.
 */
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheRight(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    transposeA: Boolean,
    bValues: DoubleArray,
    bOffset: Int,
    bRows: Int,
    bColumns: Int,
    bLeadingDimension: Int,
    transposeB: Boolean,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    m: Int,
    arrays: Scratch?,
) {
    if (transposeB) {
        arrays.borrow(bRows * bColumns) { packed ->
            for (j in 0 until bColumns) {
                for (i in 0 until bRows) packed[j + i * bColumns] = bValues[bOffset + i + j * bLeadingDimension]
            }
            multiplyFromTheRightColumns(
                kernels,
                alpha,
                aValues,
                aPointers,
                aIndices,
                aColumns,
                transposeA,
                packed,
                0,
                m,
                cValues,
                cOffset,
                cLeadingDimension,
                m,
            )
        }
    } else {
        multiplyFromTheRightColumns(
            kernels,
            alpha,
            aValues,
            aPointers,
            aIndices,
            aColumns,
            transposeA,
            bValues,
            bOffset,
            bLeadingDimension,
            cValues,
            cOffset,
            cLeadingDimension,
            m,
        )
    }
}

/** One whole contiguous dense column update per stored sparse entry, and therefore the selected Level 1 `axpy`. */
@Suppress("LongParameterList") // the operands, their flags, and both dense windows
private fun multiplyFromTheRightColumns(
    kernels: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    rows: Int,
) {
    val rowIndices = aIndices
    val values = aValues
    forEachSparseColumn(aPointers, aColumns) { column, from, to ->
        for (position in from until to) {
            val row = rowIndices[position]
            val cColumn = if (transposeA) row else column
            val bColumn = if (transposeA) column else row
            kernels.sparseEvaluatedAxpy(
                cValues,
                cOffset + cColumn * cLeadingDimension,
                alpha * values[position],
                b,
                bOffset + bColumn * ldb,
                rows,
            )
        }
    }
}

/**
 * A sparse symmetric operand on the right, whose every selected entry updates one whole contiguous dense
 * column and, off the diagonal, the mirrored one too.
 */
@Suppress("LongParameterList") // Two operands, a destination and the stored triangle.
internal fun multiplySymmetricFromTheRight(
    engine: KernelProvider,
    alpha: Double,
    aValues: DoubleArray,
    aPointers: IntArray,
    aIndices: IntArray,
    aColumns: Int,
    bValues: DoubleArray,
    bOffset: Int,
    bRows: Int,
    bLeadingDimension: Int,
    cValues: DoubleArray,
    cOffset: Int,
    cLeadingDimension: Int,
    lower: Boolean,
) {
    val rows = bRows
    val ldb = bLeadingDimension
    val ldc = cLeadingDimension
    val rowIndices = aIndices
    val values = aValues
    forEachSparseColumn(aPointers, aColumns) { column, from, to ->
        for (position in from until to) {
            val row = rowIndices[position]
            if (if (lower) row >= column else row <= column) {
                val coefficient = alpha * values[position]
                engine.sparseEvaluatedAxpy(
                    cValues,
                    cOffset + row * ldc,
                    coefficient,
                    bValues,
                    bOffset + column * ldb,
                    rows,
                )
                if (row != column) {
                    engine.sparseEvaluatedAxpy(
                        cValues,
                        cOffset + column * ldc,
                        coefficient,
                        bValues,
                        bOffset + row * ldb,
                        rows,
                    )
                }
            }
        }
    }
}

/**
 * `y += alpha · x`, evaluating even zero [alpha] because it represents a stored sparse position.
 * Level 1 `axpy` skips a zero multiplier, so that case requires its own loop.
 */
@Suppress("LongParameterList") // the two windows and the run length
private fun KernelProvider.sparseEvaluatedAxpy(
    y: DoubleArray,
    yOffset: Int,
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    length: Int,
) {
    if (alpha != 0.0) {
        this.axpy(y, yOffset, alpha, x, xOffset, length)
        return
    }
    scalarEvaluatedAxpy(y, yOffset, alpha, x, xOffset, length)
}
