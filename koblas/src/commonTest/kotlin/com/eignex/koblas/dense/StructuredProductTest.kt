package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.poisonedSymmetric
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

// Shapes are derived from the resolved backend rather than written down: a fixed extent takes the packed
// route on a narrow tile and the unpacked one on a wide machine, so every shape checks which it got.
class StructuredProductTest {
    private val rng = Random(20261102)

    /** Small enough that no backend packs it, which is the route over the operands where they lie. */
    private val directOrder = 5
    private val directDepth = 3

    @Test
    fun `the blocked and direct shapes really take the routes they are chosen for`() = withProducts { products ->
        val order = blockedOrder(products)
        val depth = blockedDepth(products, order)

        assertTrue(
            packsWindow(products, order, order, depth, OutputTriangle.Lower),
            "${products.name}: ${order}x$depth was meant to be the packed route and is not",
        )
        assertTrue(
            !packsWindow(products, directOrder, directOrder, directDepth, OutputTriangle.Lower),
            "${products.name}: ${directOrder}x$directDepth was meant to be the unpacked route and is not",
        )
        // An order one past a whole number of tiles is what leaves a block straddling the diagonal with an
        // edge in it, which is the case the selected writeback exists for.
        assertTrue(order % products.tileRows != 0, "${products.name}: the blocked order leaves no row edge")
    }

    @Test
    fun `gemmt agrees with the oracle on both routes in every orientation`() = withDenseBlas { blas ->
        for ((order, depth) in shapes()) {
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(false, true)) {
                        val a = randomMatrix(if (transposeA) depth else order, if (transposeA) order else depth, rng)
                        val b = randomMatrix(if (transposeB) order else depth, if (transposeB) depth else order, rng)
                        val start = randomMatrix(order, order, rng)
                        val expected = start.copyOf()
                        val actual = start.copyOf()
                        ReferenceBlas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, expected, lower)

                        blas.gemmt(0.875, a, transposeA, b, transposeB, -0.25, actual, lower)

                        assertClose(
                            expected.values,
                            actual.values,
                            "gemmt ${order}x$depth tA=$transposeA tB=$transposeB lower=$lower",
                            TOLERANCE,
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `the rank updates agree with the oracle on both routes in every orientation`() = withDenseBlas { blas ->
        for ((order, depth) in shapes()) {
            for (transpose in booleanArrayOf(false, true)) {
                for (lower in booleanArrayOf(false, true)) {
                    val a = randomMatrix(if (transpose) depth else order, if (transpose) order else depth, rng)
                    val b = randomMatrix(a.rows, a.cols, rng)
                    val start = randomMatrix(order, order, rng)
                    val context = "${order}x$depth transpose=$transpose lower=$lower"

                    val syrk = start.copyOf()
                    val syrkExpected = start.copyOf()
                    ReferenceBlas.syrk(0.875, a, transpose, -0.25, syrkExpected, lower)
                    blas.syrk(0.875, a, transpose, -0.25, syrk, lower)
                    assertClose(syrkExpected.values, syrk.values, "syrk $context", TOLERANCE)

                    val syr2k = start.copyOf()
                    val syr2kExpected = start.copyOf()
                    ReferenceBlas.syr2k(0.875, a, b, transpose, -0.25, syr2kExpected, lower)
                    blas.syr2k(0.875, a, b, transpose, -0.25, syr2k, lower)
                    assertClose(syr2kExpected.values, syr2k.values, "syr2k $context", TOLERANCE)
                }
            }
        }
    }

    // A NaN outside the selection catches both a write outside it and a read of it; both routes, since the
    // packed one masks at the tile and the unpacked one shortens each destination column.
    @Test
    fun `a selected-triangle product leaves the opposite triangle exactly as it found it`() = withDenseBlas { blas ->
        for ((order, depth) in shapes()) {
            for (lower in booleanArrayOf(false, true)) {
                val a = randomMatrix(order, depth, rng)
                val b = randomMatrix(depth, order, rng)
                val c = DenseMatrix.wrap(order, order, DoubleArray(order * order) { Double.NaN })

                blas.gemmt(0.875, a, false, b, false, 0.0, c, lower)

                for (column in 0 until order) {
                    for (row in 0 until order) {
                        val selected = if (lower) row >= column else row <= column
                        val context = "gemmt ${order}x$depth lower=$lower at ($row, $column)"
                        if (selected) {
                            assertTrue(c[row, column].isFinite(), "$context was not written")
                        } else {
                            assertTrue(c[row, column].isNaN(), "$context was written")
                        }
                    }
                }
            }
        }
    }

    // The shape is grown until it is packed and then pushed past one cache block of depth, and the blocks
    // the schedule produced are checked first: on a wider tile the depth would never be cut at all.
    @Test
    fun `beta reaches a selected entry once however the depth is cut`() = withProducts { products ->
        val order = blockedOrder(products)
        val depth = maxOf(blockedDepth(products, order), PRODUCT_BLOCK_DEPTH + 7)
        assertTrue(depth > PRODUCT_BLOCK_DEPTH, "the fixture does not cut the depth")
        assertTrue(
            packsWindow(products, order, order, depth, OutputTriangle.Lower),
            "${products.name}: ${order}x$depth does not take the blocked route",
        )
        for (lower in booleanArrayOf(false, true)) {
            val recorder = RecordingProducts(products)
            val blas = PortableDenseBlas(ScalarVectorKernels, panelsFor(products), recorder)
            val a = randomMatrix(order, depth, rng)
            val b = randomMatrix(depth, order, rng)
            val poisoned = DenseMatrix.wrap(order, order, DoubleArray(order * order) { Double.NaN })

            blas.gemmt(0.875, a, false, b, false, 0.0, poisoned, lower)

            val direct = recorder.blocks.filter { it.into === poisoned.values }.groupBy { it.destination }
            assertTrue(direct.isNotEmpty(), "${products.name}: no block reached the destination")
            assertTrue(
                direct.values.any { it.size > 1 },
                "${products.name}: no destination window was visited by more than one depth block",
            )
            for ((window, blocks) in direct) {
                val scaled = blocks.filter { it.beta != 1.0 }
                assertEquals(1, scaled.size, "${products.name}: window $window took beta ${scaled.size} times")
                assertEquals(depth, blocks.sumOf { it.depth }, "${products.name}: window $window lost depth")
            }
            for (column in 0 until order) {
                for (row in 0 until order) {
                    val context = "${products.name} lower=$lower at ($row, $column)"
                    if (if (lower) row >= column else row <= column) {
                        assertTrue(poisoned[row, column].isFinite(), "a zero beta left the poison $context")
                    } else {
                        assertTrue(poisoned[row, column].isNaN(), "the unselected triangle was written $context")
                    }
                }
            }

            val start = randomMatrix(order, order, rng)
            val expected = start.copyOf()
            val actual = start.copyOf()
            ReferenceBlas.gemmt(0.875, a, false, b, false, -0.25, expected, lower)
            blas.gemmt(0.875, a, false, b, false, -0.25, actual, lower)
            assertClose(expected.values, actual.values, "gemmt over several depth blocks lower=$lower", TOLERANCE)
        }
    }

    // The staged copy is what the packers read, so an order small enough to take the unpacked route would
    // leave the packed one unchecked against the alias it is allowed to have.
    @Test
    fun `a packed rank update over its own destination agrees with the oracle`() = withProducts { products ->
        val order = packedSquareOrder(products)
        assertTrue(
            packsWindow(products, order, order, order, OutputTriangle.Lower),
            "${products.name}: a square of $order does not take the blocked route",
        )
        val blas = PortableDenseBlas(ScalarVectorKernels, panelsFor(products), products)
        val start = randomMatrix(order, order, rng)
        for (lower in booleanArrayOf(false, true)) {
            val expected = start.copyOf()
            ReferenceBlas.syrk(0.875, start.copyOf(), false, -0.25, expected, lower)
            val aliased = start.copyOf()

            blas.syrk(0.875, aliased, false, -0.25, aliased, lower)

            assertClose(
                expected.values,
                aliased.values,
                "${products.name}: packed syrk into its own operand lower=$lower",
                TOLERANCE,
            )
        }
    }

    // The two sums at the entry below are zero and one, so an infinite multiplier tells composing (`∞·0 +
    // ∞·1`, a NaN, which is promised) from fusing (`∞·(0 + 1)`, an infinity).
    @Test
    fun `a rank two-k update scales each of its two products separately`() = withDenseBlas { blas ->
        val a = DenseMatrix.wrap(2, 1, doubleArrayOf(1.0, 0.0))
        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 1.0))
        val c = DenseMatrix.wrap(2, 2, DoubleArray(4))

        blas.syr2k(Double.POSITIVE_INFINITY, a, b, transpose = false, beta = 0.0, c = c, lower = true)

        assertTrue(
            c[1, 0].isNaN(),
            "the two products were not scaled separately; a fused sum would leave ${c[1, 0]}",
        )
    }

    // Offsets and leading dimensions past the extents are how a structured algorithm hands over a strip, and
    // every entry outside the window carries a guard, including the rows between one column and the next.
    @Test
    fun `a product window writes only inside the offsets and the triangle it was given`() = withProducts { products ->
        for (selected in listOf(OutputTriangle.Full, OutputTriangle.Lower, OutputTriangle.Upper)) {
            for (packed in booleanArrayOf(false, true)) {
                for (transposeA in booleanArrayOf(false, true)) {
                    assertWindowed(products, selected, packed, transposeA, transposeB = !transposeA)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the backend and the four facts that pick out one case of the sweep
    private fun assertWindowed(
        products: DenseProductKernels,
        selected: OutputTriangle,
        packed: Boolean,
        transposeA: Boolean,
        transposeB: Boolean,
    ) {
        val order = if (packed) blockedOrder(products) else directOrder
        val depth = if (packed) blockedDepth(products, order) else directDepth
        val context = "${products.name} ${order}x$depth $selected packed=$packed tA=$transposeA tB=$transposeB"
        assertEquals(
            packed,
            packsWindow(products, order, order, depth, selected),
            "$context: the fixture did not take the route it was chosen for",
        )
        val a = randomMatrix(if (transposeA) depth else order, if (transposeA) order else depth, rng)
        val b = randomMatrix(if (transposeB) order else depth, if (transposeB) depth else order, rng)
        val start = randomMatrix(order, order, rng)
        val expected = start.copyOf()
        if (selected == OutputTriangle.Full) {
            ReferenceBlas.gemm(ALPHA, a, transposeA, b, transposeB, BETA, expected)
        } else {
            ReferenceBlas.gemmt(
                ALPHA,
                a,
                transposeA,
                b,
                transposeB,
                BETA,
                expected,
                selected == OutputTriangle.Lower,
            )
        }

        val left = padded(a, A_ORIGIN, a.rows + 3)
        val right = padded(b, B_ORIGIN, b.rows + 5)
        val destination = padded(start, C_ORIGIN, order + 7)
        val leftBefore = left.copyOf()
        val rightBefore = right.copyOf()

        productWindow(
            products, panelsFor(products), ALPHA,
            left, A_ORIGIN, a.rows + 3, transposeA,
            right, B_ORIGIN, b.rows + 5, transposeB,
            BETA, destination, C_ORIGIN, order + 7,
            order, order, depth, selected, Workspace(),
        )

        assertContentEquals(leftBefore, left, "$context: the left operand was written")
        assertContentEquals(rightBefore, right, "$context: the right operand was written")
        for (index in destination.indices) {
            val local = index - C_ORIGIN
            val column = local / (order + 7)
            val row = local - column * (order + 7)
            val inside = local >= 0 && column < order && row in 0 until order
            val written = inside && !outsideTriangle(row, 1, column, 1, selected)
            if (written) {
                assertClose(
                    expected[row, column],
                    destination[index],
                    "$context at ($row, $column)",
                    TOLERANCE,
                )
            } else if (inside) {
                assertEquals(start[row, column], destination[index], "$context wrote ($row, $column)")
            } else {
                assertEquals(GUARD, destination[index], "$context wrote outside its window at $index")
            }
        }
    }

    /** [a] laid out at [origin] with [leading] rows between columns, everything else a guard value. */
    private fun padded(a: DenseMatrix, origin: Int, leading: Int): DoubleArray {
        val values = DoubleArray(origin + a.cols * leading + origin) { GUARD }
        for (column in 0 until a.cols) {
            for (row in 0 until a.rows) values[origin + row + column * leading] = a[row, column]
        }
        return values
    }

    @Test
    fun `symm agrees with the oracle on both sides and reads only the stored triangle`() = withDenseBlas { blas ->
        for ((order, sides) in symmetricShapes()) {
            for (lower in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    val (full, poisoned) = poisonedSymmetric(rng, order, lower)
                    val rows = if (right) sides else order
                    val columns = if (right) order else sides
                    val b = randomMatrix(rows, columns, rng)
                    val start = randomMatrix(rows, columns, rng)
                    val expected = start.copyOf()
                    val actual = start.copyOf()
                    // The oracle takes the full symmetric matrix, so its answer does not depend on which
                    // half this call was given.
                    ReferenceBlas.symm(0.875, full, b, -0.25, expected, lower = true, right = right)

                    blas.symm(0.875, poisoned, b, -0.25, actual, lower, right)

                    assertClose(
                        expected.values,
                        actual.values,
                        "symm order=$order sides=$sides lower=$lower right=$right",
                        TOLERANCE,
                    )
                }
            }
        }
    }

    /** A symmetric product with nothing to add still overwrites its destination rather than scaling it. */
    @Test
    fun `symm with a zero beta overwrites a poisoned destination`() = withDenseBlas { blas ->
        val order = SYMMETRIC_BLOCK + 3
        val sides = 5
        val (_, poisoned) = poisonedSymmetric(rng, order, lower = true)
        val b = randomMatrix(order, sides, rng)
        val c = DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { Double.NaN })

        blas.symm(0.875, poisoned, b, 0.0, c, lower = true)

        assertTrue(c.values.all { it.isFinite() }, "a zero beta kept the poison in the destination")
    }

    /** A rank update whose operand is its own destination reads the operand it was given, not the result. */
    @Test
    fun `a rank update over its own destination agrees with the oracle`() = withDenseBlas { blas ->
        val order = 2 * PORTABLE_PRODUCT_TILE + 1
        val start = randomMatrix(order, order, rng)
        for (lower in booleanArrayOf(false, true)) {
            val expected = start.copyOf()
            ReferenceBlas.syrk(0.875, start.copyOf(), false, -0.25, expected, lower)
            val aliased = start.copyOf()

            blas.syrk(0.875, aliased, false, -0.25, aliased, lower)

            assertClose(expected.values, aliased.values, "syrk into its own operand lower=$lower", TOLERANCE)
        }
    }

    /** A symmetric product whose dense operand is its destination reads what the caller supplied. */
    @Test
    fun `symm over its own destination agrees with the oracle`() = withDenseBlas { blas ->
        val order = SYMMETRIC_BLOCK + 3
        val (full, poisoned) = poisonedSymmetric(rng, order, lower = true)
        val start = randomMatrix(order, order, rng)
        val expected = start.copyOf()
        ReferenceBlas.symm(0.875, full, start.copyOf(), -0.25, expected, lower = true)
        val aliased = start.copyOf()

        blas.symm(0.875, poisoned, aliased, -0.25, aliased, lower = true)

        assertClose(expected.values, aliased.values, "symm into its own dense operand", TOLERANCE)
    }

    // Every operand shares one empty array, so a call testing for an alias before testing for an empty
    // destination would take a staging loan for a result that does not exist.
    @Test
    fun `a structured call with no output borrows nothing`() = withDenseBlas { blas ->
        val workspace = Workspace()
        val shared = DoubleArray(0)
        val tall = DenseMatrix.wrap(0, 4, shared)
        val wide = DenseMatrix.wrap(4, 0, shared)
        val square = DenseMatrix.wrap(0, 0, shared)
        val destination = DenseMatrix.wrap(0, 4, shared)

        blas.gemmt(ALPHA, tall, false, wide, false, BETA, square, workspace = workspace)
        blas.syrk(ALPHA, tall, false, BETA, square, workspace = workspace)
        blas.syr2k(ALPHA, tall, tall, false, BETA, square, workspace = workspace)
        blas.symm(ALPHA, square, tall, BETA, destination, workspace = workspace)
        blas.trsm(square, tall, lower = true, workspace = workspace)
        blas.trmm(square, tall, lower = true, workspace = workspace)

        assertEquals(0, workspace.idleLengths(), "a call with no output borrowed scratch it had no use for")
    }

    // Run under a recorder, so the claim is checked against the blocks the schedule produced.
    @Test
    fun `a selected-triangle route names the blocks the call really cut`() = withProducts { products ->
        val panels = panelsFor(products)
        val order = blockedOrder(products)
        val depth = blockedDepth(products, order)
        for (lower in booleanArrayOf(false, true)) {
            assertTriangleProductRouteNamesExecutedBlocks(products, panels, order, depth, lower)
            assertTriangleProductRouteNamesExecutedBlocks(products, panels, directOrder, directDepth, lower)
            assertTriangleProductRouteNamesExecutedBlocks(
                products,
                panels,
                order,
                depth,
                lower,
                transposeA = true,
                transposeB = true,
            )
        }
    }

    @Test
    @OptIn(KoblasEngineApi::class)
    fun `a rank two-k route reports the composition it is`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            val route = engine.routeOf(
                DenseMatrixOperation.Syr2k,
                DenseCall(64, 64, 0.875, -0.25, depth = 64),
            )

            assertEquals(DenseMatrixOperation.Syr2k, route.operation)
            assertContains(assertNotNull(route.reason), "composed rather than fused")
            assertTrue(!route.exactlyMeasurable, route.toString())
        }
    }

    @OptIn(KoblasEngineApi::class)
    private fun koblasPanels(): DensePanelKernels = BuiltinEngines.simd?.panelKernels ?: PortablePanelKernels

    /** The panel backend that belongs beside [products], so a composition under test is a real one. */
    private fun panelsFor(products: DenseProductKernels): DensePanelKernels =
        if (products === PortableProductKernels) PortablePanelKernels else koblasPanels()

    /** The blocked shape and the direct one, so every conformance check covers both routes. */
    private fun shapes(): List<Pair<Int, Int>> {
        val products = defaultProducts()
        val order = blockedOrder(products)
        return listOf(order to blockedDepth(products, order), directOrder to directDepth)
    }

    /** The same for a symmetric operand, whose blocks are the order's rather than the destination's. */
    private fun symmetricShapes(): List<Pair<Int, Int>> = listOf(SYMMETRIC_BLOCK + 3 to 7, directOrder to 2, 1 to 3)

    private companion object {
        const val TOLERANCE = 1e-9
        const val ALPHA = 0.875
        const val BETA = -0.25

        /** A value no fixture produces, so an entry still holding it was not written. */
        const val GUARD = -7.5

        /** Origins past the start of each array, so nothing lands there by starting from zero. */
        const val A_ORIGIN = 3
        const val B_ORIGIN = 11
        const val C_ORIGIN = 5
    }
}

/** Runs [body] against every product backend this platform has, which is what decides the block geometry. */
@OptIn(KoblasEngineApi::class)
internal fun withProducts(body: (DenseProductKernels) -> Unit) {
    body(PortableProductKernels)
    BuiltinEngines.simd?.let { body(it.productKernels) }
}

/** The product backend of the engine whose scheduling the conformance checks above run under. */
@OptIn(KoblasEngineApi::class)
internal fun defaultProducts(): DenseProductKernels = BuiltinEngines.simd?.productKernels ?: PortableProductKernels

/**
 * A destination order past several of this backend's tiles and one row short of a whole number of them,
 * read from the backend because a fixed order fills a narrow tile exactly and leaves a wide one short.
 */
internal fun blockedOrder(products: DenseProductKernels): Int = 4 * maxOf(products.tileRows, products.tileColumns) + 1

/** The smallest depth at which a triangle-selected product of this order is packed into tiles. */
internal fun blockedDepth(products: DenseProductKernels, order: Int): Int {
    var depth = 8
    while (depth < MAXIMUM_FIXTURE_DEPTH && !packsWindow(products, order, order, depth, OutputTriangle.Lower)) {
        depth *= 2
    }
    return depth
}

/** A bound on the fixtures above, so a backend that never packs produces a test that fails rather than hangs. */
private const val MAXIMUM_FIXTURE_DEPTH = 4096

/**
 * A square order at which a product of it by itself is packed and which leaves a row edge; a rank update's
 * operand can share its destination only at one shape, so the packed alias case cannot use a rectangle.
 */
internal fun packedSquareOrder(products: DenseProductKernels): Int {
    var order = blockedOrder(products)
    while (order < MAXIMUM_FIXTURE_ORDER && !packsWindow(products, order, order, order, OutputTriangle.Lower)) {
        order += products.tileRows
    }
    return order
}

/** A bound on the search above, for the reason [MAXIMUM_FIXTURE_DEPTH] is one. */
private const val MAXIMUM_FIXTURE_ORDER = 512
