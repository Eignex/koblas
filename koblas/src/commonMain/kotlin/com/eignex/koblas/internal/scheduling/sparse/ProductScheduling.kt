@file:Suppress("FunctionParameterNaming", "VariableNaming") // Independent windows and extents.

package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.dense.scalarEvaluatedAxpy
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.operations.staged
import com.eignex.koblas.internal.scheduling.forEachPanel
import com.eignex.koblas.internal.storage.borrowOptional
import kotlin.jvm.JvmInline
import kotlin.math.min

/**
 * The most right-hand sides the sparse scheduling will hold staged adjacent at once.
 *
 * A ceiling over the backend's recommendation rather than the width itself: how many right-hand sides a
 * panel wants is the backend's question, since it is the one with the body, and how much of a dense
 * block may be copied into a scratch buffer while one sparse column is walked is the scheduling's.
 * Where the recommendation is below this, this changes nothing.
 */
private const val CONTIGUOUS_RHS_PANEL: Int = 64

/**
 * Stored entries per dense element copied, below which a sparse product keeps the caller's layout.
 *
 * Staging a panel of right-hand sides into adjacent order costs a copy of the dense operand, or of the
 * destination twice over, and buys vector arithmetic over the entries the traversal walks. The ratio
 * between those two is this number, and a product too sparse to pay it runs where its operands already
 * are.
 *
 * The measured ratio is four: the band just below it was no better than the caller's own layout,
 * and above it the copy pays and keeps paying as the support thickens. A gathering product copies half
 * as much as a scattering one, so the same ratio admits it at half the density, which is what a ratio
 * rather than a density is for.
 */
private const val STAGED_RHS_CROSSOVER: Int = 4

/**
 * Chooses right-hand-side grouping and staging for a sparse product with a dense block.
 *
 * Staging makes right-hand sides adjacent for vector panels. The backend recommends a width for the
 * resulting layout; [CONTIGUOUS_RHS_PANEL] caps retained staging storage. Copy only when the
 * backend prefers adjacent panels and the accelerated arithmetic outweighs the copy. Already-adjacent
 * inputs need no copy, and the portable backend never stages.
 *
 * [RhsPlan] encodes staging in the width's sign without allocating a plan object.
 *
 * @param kernels backend sparse panel kernels.
 * @param rows destination rows used to choose grouping.
 * @param sides right-hand-side count.
 * @param entries stored entries traversed per right-hand side.
 * @param copiedPerSide elements copied per staged right-hand side, counting a destination twice for
 *   read and write. `Long` prevents valid extents from overflowing the copy-cost calculation.
 * @param nativelyContiguous whether right-hand sides are already adjacent.
 * @param reduction whether panels reduce into accumulators or update destination windows.
 */
@Suppress("LongParameterList") // the seam, the two extents, the copy's cost and what the direction admits
internal fun planRightHandSides(
    kernels: KoblasEngine,
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
 * The plan a sparse-by-dense product takes.
 *
 * What a staged copy pays for is what the direction reads: a transposed operand reduces a column into an
 * accumulator of its own, so the copy is the dense source and one already in right-hand-side order needs
 * none; an untransposed one spreads a column across the destination, which is read in and written back.
 */
internal fun productRhsPlan(
    kernels: KoblasEngine,
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
internal fun symmetricRhsPlan(kernels: KoblasEngine, order: Int, sides: Int, entries: Int): RhsPlan =
    planRightHandSides(kernels, order, sides, entries, copiedPerSide = 3L * order, nativelyContiguous = false)

/** Right-hand-side grouping, with a negative encoding when the dense block needs staging. */
@JvmInline
internal value class RhsPlan(private val encoded: Int) {
    val width: Int get() = if (encoded < 0) -encoded else encoded
    val staged: Boolean get() = encoded < 0
}

/**
 * Copies [actual] right-hand sides of a dense block into [panel], or back out of it when not [toPanel].
 *
 * Right-hand side `r` at index `i` of the block is `values(origin + r · rhsStride + i · indexStride)`, and
 * [count] is how far the indexed axis runs. The panel holds [width] adjacent right-hand sides per index,
 * which is the layout whose right-hand-side axis a vector body can load. Writing back covers every position,
 * including one the arithmetic never reached: a destination is staged by copying it rather than by starting
 * from zero, so an untouched entry returns bit for bit and a negative zero stays negative.
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
 * `C += alpha · op(A) · op(B)` for a sparse `A` on the left, reusing each walk over a panel of right-hand
 * sides.
 *
 * The transposed operand reduces the rows its column stores into one output row, so its accumulator is
 * adjacent whatever the dense block looks like and only the block's own layout decides whether the panel
 * vectorises. The untransposed one spreads one inner index across the rows it stores, so its destination is
 * the dense block and a column-major one is strided there; that is the direction a staged panel is for.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and the shape already worked out
internal fun multiplyFromTheLeft(
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    n: Int,
    k: Int,
    arrays: Scratch?,
) {
    // Return before staging or traversing: an empty axis may be paired with an enormous one.
    if (m == 0 || n == 0 || k == 0) return
    val ldb = b.leadingDimension
    val bRhsStride = if (transposeB) 1 else ldb
    val bIndexStride = if (transposeB) ldb else 1
    val plan = productRhsPlan(kernels, m, n, k, a.nnz, transposeA, transposeB)
    val width = plan.width
    // Single-right-hand-side groups reach no panel; avoid retaining unused scratch in the arrays.
    arrays.borrowOptional(if (width > 1 && n > 1) width else 0) { work ->
        arrays.borrowOptional(if (plan.staged) width * (if (transposeA) k else m) else 0) { panel ->
            forEachPanel(n, width) { columnStart, actual ->
                if (transposeA) {
                    gatherPanel(
                        kernels, alpha, a, b, c, k, columnStart, actual, width, plan.staged,
                        bRhsStride, bIndexStride, panel, work,
                    )
                } else {
                    scatterPanel(
                        kernels, alpha, a, b, c, m, columnStart, actual, width, plan.staged,
                        bRhsStride, bIndexStride, panel, work,
                    )
                }
            }
        }
    }
}

/** One panel of right-hand sides of `C += alpha · Aᵀ · op(B)`, staged or in the caller's own layout. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun gatherPanel(
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
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
    val origin = b.offset + columnStart * bRhsStride
    val cOrigin = c.offset + columnStart * c.leadingDimension
    withRhsPanel(
        b.values, origin, bRhsStride, bIndexStride, k, actual, staged, panel, width, writeBack = false,
    ) { source, offset, rhsStride, indexStride ->
        if (actual == 1) {
            forEachSparseColumn(a) { outputRow, from, to ->
                sparseGatherProductColumn(
                    alpha, a.rowIndices, a.values, from, to, source, offset, indexStride, c.values, cOrigin + outputRow,
                )
            }
        } else {
            forEachSparseColumn(a) { outputRow, from, to ->
                kernels.sparseGatherProductPanel(
                    alpha, a.rowIndices, a.values, from, to,
                    source, offset, rhsStride, indexStride, actual,
                    c.values, cOrigin + outputRow, c.leadingDimension, work,
                )
            }
        }
    }
}

/** One panel of right-hand sides of `C += alpha · A · op(B)`, with the destination staged or in place. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the panel window, and the staged buffers
private fun scatterPanel(
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
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
    val origin = c.offset + columnStart * c.leadingDimension
    val bOrigin = b.offset + columnStart * bRhsStride
    withRhsPanel(
        c.values, origin, c.leadingDimension, 1, m, actual, staged, panel, width, writeBack = true,
    ) { destination, offset, rhsStride, indexStride ->
        if (actual == 1) {
            forEachSparseColumn(a) { inner, from, to ->
                sparseScatterProductColumn(
                    alpha, a.rowIndices, a.values, from, to,
                    b.values, bOrigin + inner * bIndexStride, destination, offset, indexStride,
                )
            }
        } else {
            forEachSparseColumn(a) { inner, from, to ->
                kernels.sparseScatterProductPanel(
                    alpha, a.rowIndices, a.values, from, to,
                    b.values, bOrigin + inner * bIndexStride, bRhsStride, actual,
                    destination, offset, rhsStride, indexStride, work,
                )
            }
        }
    }
}

/**
 * `C += alpha · A · B` for a symmetric sparse `A` on the left, over panels of right-hand sides.
 *
 * Both dense operands are the same shape and the same layout, so one panel geometry serves the source and
 * the destination. Staging copies both, and writes only the destination back; that is three passes over a
 * right-hand side against the two a general product's destination costs, which is why the two callers weigh
 * the copy differently.
 */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, the triangle flag, and the scratch
internal fun multiplySymmetricFromTheLeft(
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    lower: Boolean,
    arrays: Scratch?,
) {
    val n = a.rows
    val sides = b.cols
    val ld = b.leadingDimension
    val ldc = c.leadingDimension
    if (n == 0 || sides == 0) return
    val plan = symmetricRhsPlan(kernels, n, sides, a.nnz)
    val width = plan.width
    val panelSize = if (plan.staged) width * n else 0
    arrays.borrowOptional(panelSize) { source ->
        arrays.borrowOptional(panelSize) { destination ->
            forEachPanel(sides, width) { columnStart, actual ->
                val bOrigin = b.offset + columnStart * ld
                val cOrigin = c.offset + columnStart * ldc
                withRhsPanel(
                    b.values, bOrigin, ld, 1, n, actual, plan.staged, source, width, writeBack = false,
                ) { sourceValues, offset, rhsStride, indexStride ->
                    withRhsPanel(
                        c.values, cOrigin, ldc, 1, n, actual, plan.staged, destination, width, writeBack = true,
                    ) { into, destinationOffset, destinationStride, _ ->
                        forEachSparseColumn(a) { column, from, to ->
                            if (actual == 1) {
                                sparseSymmetricLeftColumnSingle(
                                    alpha, column, a.rowIndices, a.values, from, to,
                                    sourceValues, into, offset, indexStride, lower, destinationOffset,
                                )
                            } else {
                                kernels.sparseSymmetricLeftColumn(
                                    alpha, column, a.rowIndices, a.values, from, to,
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
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DenseMatrix,
    transposeB: Boolean,
    c: DenseMatrix,
    m: Int,
    arrays: Scratch?,
) {
    if (transposeB) {
        arrays.borrow(DenseMatrix.entryCount(b.rows, b.cols)) { packed ->
            for (j in 0 until b.cols) {
                for (i in 0 until b.rows) packed[j + i * b.cols] = b.values[b.offset + i + j * b.leadingDimension]
            }
            multiplyFromTheRightColumns(kernels, alpha, a, transposeA, packed, 0, m, c, m)
        }
    } else {
        multiplyFromTheRightColumns(kernels, alpha, a, transposeA, b.values, b.offset, b.leadingDimension, c, m)
    }
}

/** One whole contiguous dense column update per stored entry of [a], and therefore the selected Level 1 `axpy`. */
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // the operands, their flags, and both dense windows
private fun multiplyFromTheRightColumns(
    kernels: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    transposeA: Boolean,
    b: DoubleArray,
    bOffset: Int,
    ldb: Int,
    c: DenseMatrix,
    rows: Int,
) {
    val rowIndices = a.rowIndices
    val values = a.values
    forEachSparseColumn(a) { column, from, to ->
        for (position in from until to) {
            val row = rowIndices[position]
            val cColumn = if (transposeA) row else column
            val bColumn = if (transposeA) column else row
            kernels.sparseEvaluatedAxpy(
                c.values,
                c.offset + cColumn * c.leadingDimension,
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
@OptIn(UnsafeKoblasApi::class)
@Suppress("LongParameterList") // Two operands, a destination and the stored triangle.
internal fun multiplySymmetricFromTheRight(
    engine: KoblasEngine,
    alpha: Double,
    a: SparseMatrix,
    b: DenseMatrix,
    c: DenseMatrix,
    lower: Boolean,
) {
    val rows = b.rows
    val ldb = b.leadingDimension
    val ldc = c.leadingDimension
    val rowIndices = a.rowIndices
    val values = a.values
    forEachSparseColumn(a) { column, from, to ->
        for (position in from until to) {
            val row = rowIndices[position]
            if (if (lower) row >= column else row <= column) {
                val coefficient = alpha * values[position]
                engine.sparseEvaluatedAxpy(
                    c.values,
                    c.offset + row * ldc,
                    coefficient,
                    b.values,
                    b.offset + column * ldb,
                    rows,
                )
                if (row != column) {
                    engine.sparseEvaluatedAxpy(
                        c.values,
                        c.offset + column * ldc,
                        coefficient,
                        b.values,
                        b.offset + row * ldb,
                        rows,
                    )
                }
            }
        }
    }
}

/**
 * `y += alpha · x` over a contiguous run, forming the product even where [alpha] is zero.
 *
 * This update represents a stored sparse position, including an explicit zero. Its participation is
 * required by the sparse structural contract, unlike a zero coefficient in a dense update. Level 1
 * `axpy` skips a zero multiplier, so that case needs its own loop here.
 */
@Suppress("LongParameterList") // the two windows and the run length
private fun KoblasEngine.sparseEvaluatedAxpy(
    y: DoubleArray,
    yOffset: Int,
    alpha: Double,
    x: DoubleArray,
    xOffset: Int,
    length: Int,
) {
    if (alpha != 0.0) {
        kernels.axpy(y, yOffset, alpha, x, xOffset, length)
        return
    }
    scalarEvaluatedAxpy(y, yOffset, alpha, x, xOffset, length)
}
