@file:Suppress("LongParameterList") // a product block carries two packed panels, its extents and its scaling

package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// The product block contract, over any implementation. Every backend has to satisfy it at every extent,
// including extents that do not fill its tile, so the assertions live here rather than beside one of them.
//
// The packed operands below are built from the index formula the contract states, written out again rather
// than taken from the production packer. A block read through the same code that wrote it would agree with
// itself whatever either of them did.

/** Guard entries around and inside the destination buffer, so an implementation that overruns fails. */
private const val GUARD = 3

/** Row counts that straddle a lane block, a tile and the boundary between them. */
private val BLOCK_ROWS = intArrayOf(1, 2, 3, 4, 5, 7, 8, 9, 16, 17)

/** Column counts, including one, a tail beside every tile width, and one wider than any of them. */
private val BLOCK_COLUMNS = intArrayOf(1, 2, 3, 4, 5, 9)

/** Depths, including none at all and one that is not a multiple of anything. */
private val BLOCK_DEPTHS = intArrayOf(0, 1, 2, 3, 5, 17)

/** The scalings that matter: overwrite, accumulate, and a general pair. */
private val SCALINGS = listOf(1.0 to 0.0, 1.0 to 1.0, 0.875 to -0.25, -1.5 to 1.0)

/** Packs a left operand from the index formula in the contract, independently of the production packer. */
internal fun packedLeftFixture(rows: Int, depth: Int, group: Int, entry: (Int, Int) -> Double): DoubleArray {
    val groups = (rows + group - 1) / group
    val panel = DoubleArray(groups * depth * group)
    for (i in 0 until rows) {
        for (p in 0 until depth) panel[i / group * (depth * group) + p * group + i % group] = entry(i, p)
    }
    return panel
}

/** Packs a right operand from the index formula in the contract, independently of the production packer. */
internal fun packedRightFixture(depth: Int, columns: Int, group: Int, entry: (Int, Int) -> Double): DoubleArray {
    val groups = (columns + group - 1) / group
    val panel = DoubleArray(groups * depth * group)
    for (j in 0 until columns) {
        for (p in 0 until depth) panel[j / group * (depth * group) + p * group + j % group] = entry(p, j)
    }
    return panel
}

/** That [kernels] computes `alpha · A · B + beta · C` over every extent and writes nowhere else. */
internal fun assertProductBlockAgreesWithReference(kernels: DenseProductKernels) {
    val rng = Random(20261003)
    for (rows in BLOCK_ROWS) {
        for (columns in BLOCK_COLUMNS) {
            for (depth in BLOCK_DEPTHS) {
                val (alpha, beta) = SCALINGS[(rows + columns + depth) % SCALINGS.size]
                assertOneProductBlock(kernels, rows, columns, depth, alpha, beta, rng)
            }
        }
    }
}

private fun assertOneProductBlock(
    kernels: DenseProductKernels,
    rows: Int,
    columns: Int,
    depth: Int,
    alpha: Double,
    beta: Double,
    rng: Random,
) {
    val left = DoubleArray(rows * depth) { rng.nextDouble(-1.0, 1.0) }
    val right = DoubleArray(depth * columns) { rng.nextDouble(-1.0, 1.0) }
    val packedA = packedLeftFixture(rows, depth, kernels.tileRows) { i, p -> left[i + p * rows] }
    val packedB = packedRightFixture(depth, columns, kernels.tileColumns) { p, j -> right[p + j * depth] }
    val ldc = rows + GUARD
    val original = DoubleArray(GUARD + ldc * columns + GUARD) { rng.nextDouble(-1.0, 1.0) }
    val c = original.copyOf()

    kernels.productBlock(
        alpha, packedA, 0, depth * kernels.tileRows, packedB, 0, depth * kernels.tileColumns,
        rows, columns, depth, beta, c, GUARD, ldc,
    )

    val context = "${kernels.name} ${rows}x${columns}x$depth alpha=$alpha beta=$beta"
    for (j in 0 until columns) {
        for (i in 0 until rows) {
            var sum = 0.0
            for (p in 0 until depth) sum += left[i + p * rows] * right[p + j * depth]
            val at = GUARD + i + j * ldc
            val expected = if (beta == 0.0) alpha * sum else alpha * sum + beta * original[at]
            assertClose(expected, c[at], "$context at ($i, $j)")
        }
    }
    assertUntouchedOutside(original, c, rows, columns, ldc, context)
}

/** Every entry the window does not select, still holding what it held. */
private fun assertUntouchedOutside(
    original: DoubleArray,
    c: DoubleArray,
    rows: Int,
    columns: Int,
    ldc: Int,
    context: String,
) {
    for (at in original.indices) {
        val inside = at >= GUARD && (at - GUARD) % ldc < rows && (at - GUARD) / ldc < columns
        if (!inside) assertEquals(original[at], c[at], "$context wrote outside its window at $at")
    }
}

/**
 * That a block reads no packed panel where its own extents say there is nothing to read.
 *
 * Handed empty arrays, so anything read at all is an index out of bounds rather than a wrong number.
 */
internal fun assertEmptyProductBlockReadsNothing(kernels: DenseProductKernels) {
    val nothing = DoubleArray(0)
    val c = DoubleArray(12) { 1.0 + it }
    kernels.productBlock(2.0, nothing, 0, 0, nothing, 0, 0, 0, 3, 5, 0.5, c, 0, 4)
    kernels.productBlock(2.0, nothing, 0, 0, nothing, 0, 0, 3, 0, 5, 0.5, c, 0, 4)
    assertEquals(DoubleArray(12) { 1.0 + it }.toList(), c.toList(), "${kernels.name} touched an empty window")

    kernels.productBlock(2.0, nothing, 0, 0, nothing, 0, 0, 3, 2, 0, 0.5, c, 0, 4)
    for (j in 0 until 2) {
        for (i in 0 until 3) {
            assertEquals(0.5 * (1.0 + i + j * 4), c[i + j * 4], "${kernels.name} depth-zero block at ($i, $j)")
        }
    }
}

/**
 * That a poisoned destination is overwritten rather than multiplied, which is what a zero beta means.
 *
 * A NaN standing in the output would survive `0.0 * NaN`, so this is the difference between the contract and
 * an arithmetic shortcut that looks like it.
 */
internal fun assertZeroBetaOverwritesPoison(kernels: DenseProductKernels) {
    val rows = kernels.tileRows + 1
    val columns = kernels.tileColumns + 1
    val depth = 3
    val packedA = packedLeftFixture(rows, depth, kernels.tileRows) { i, p -> 1.0 + i + p }
    val packedB = packedRightFixture(depth, columns, kernels.tileColumns) { p, j -> 1.0 + p - j }
    val ldc = rows
    val c = DoubleArray(ldc * columns) { if (it % 2 == 0) Double.NaN else Double.POSITIVE_INFINITY }

    kernels.productBlock(
        0.5, packedA, 0, depth * kernels.tileRows, packedB, 0, depth * kernels.tileColumns,
        rows, columns, depth, 0.0, c, 0, ldc,
    )

    for (j in 0 until columns) {
        for (i in 0 until rows) {
            var sum = 0.0
            for (p in 0 until depth) sum += (1.0 + i + p) * (1.0 + p - j)
            assertClose(0.5 * sum, c[i + j * ldc], "${kernels.name} poisoned output at ($i, $j)")
        }
    }
}

/**
 * That a retained panel is read in depth slices, which is what the group stride is for.
 *
 * The panels here are packed over the whole shared dimension once, as a retained operand is, and the block
 * reads a window of it. Accumulating the slices in turn has to reach the same product as one whole block,
 * with beta carried by the first of them.
 */
internal fun assertDepthSlicesAccumulate(kernels: DenseProductKernels) {
    val rng = Random(20261004)
    val rows = kernels.tileRows + 3
    val columns = kernels.tileColumns + 3
    val depth = 11
    val left = DoubleArray(rows * depth) { rng.nextDouble(-1.0, 1.0) }
    val right = DoubleArray(depth * columns) { rng.nextDouble(-1.0, 1.0) }
    val packedA = packedLeftFixture(rows, depth, kernels.tileRows) { i, p -> left[i + p * rows] }
    val packedB = packedRightFixture(depth, columns, kernels.tileColumns) { p, j -> right[p + j * depth] }
    val aStride = depth * kernels.tileRows
    val bStride = depth * kernels.tileColumns
    val original = DoubleArray(rows * columns) { rng.nextDouble(-1.0, 1.0) }
    val whole = original.copyOf()
    val sliced = original.copyOf()

    kernels.productBlock(
        0.875, packedA, 0, aStride, packedB, 0, bStride, rows, columns, depth, -0.25, whole, 0, rows,
    )
    var step = 0
    val cuts = intArrayOf(4, 1, 6)
    for (cut in cuts) {
        kernels.productBlock(
            0.875, packedA, step * kernels.tileRows, aStride, packedB, step * kernels.tileColumns, bStride,
            rows, columns, cut, if (step == 0) -0.25 else 1.0, sliced, 0, rows,
        )
        step += cut
    }
    assertEquals(depth, step, "the cuts must cover the depth")
    for (at in whole.indices) {
        assertTrue(
            abs(whole[at] - sliced[at]) <= 1e-12 * maxOf(1.0, abs(whole[at])),
            "${kernels.name} depth slices differ at $at: ${whole[at]} vs ${sliced[at]}",
        )
    }
}
