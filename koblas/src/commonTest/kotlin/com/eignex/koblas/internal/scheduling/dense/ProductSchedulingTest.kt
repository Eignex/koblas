package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.RecordingProducts
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.dense.assertTriangleProductStaysInTriangle
import com.eignex.koblas.dense.blockedDepth
import com.eignex.koblas.dense.blockedOrder
import com.eignex.koblas.dense.packedSquareOrder
import com.eignex.koblas.dense.withEngines
import com.eignex.koblas.dense.withProductKernels
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.engine.panelProvider
import com.eignex.koblas.internal.engine.productProvider
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.PACKED_PRODUCT_MINIMUM_WORK
import com.eignex.koblas.internal.kernels.packsProductByWork
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.kernels.scalar.dense.SCALAR_PRODUCT_TILE
import com.eignex.koblas.internal.operations.staged
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// The shapes reach what block scheduling has that a small product does not: more than one cache block on
// each axis, a depth cut into several blocks, and extents that fill no tile exactly. They are derived from
// the resolved backend rather than written down: a fixed extent takes the packed route on a narrow tile and
// the unpacked one on a wide machine, so every shape checks which it got.
class ProductSchedulingTest {
    private val rng = Random(20261102)

    /** Small enough that no backend packs it, which is the route over the operands where they lie. */
    private val directOrder = 5
    private val directDepth = 3

    /**
     * Shapes cross each block boundary while keeping the other axes small for the oracle.
     * Tile-derived extents ensure packing on wide species and leave remainder tiles.
     */
    private fun blockedShapes(tile: EngineKernels): List<Triple<Int, Int, Int>> {
        val rows = packedExtent(tile.tileRows)
        val columns = packedExtent(tile.tileColumns)
        return listOf(
            Triple(rows, columns, PRODUCT_BLOCK_DEPTH * 2 + 17),
            Triple(PRODUCT_BLOCK_ROWS * 2 + 7, columns, 13),
            Triple(rows, PRODUCT_BLOCK_COLUMNS * 2 + 5, 13),
            Triple(rows + 13, columns + 5, 41),
        )
    }

    /** More than one whole [tile] and not a multiple of it, and enough of them to be worth packing. */
    private fun packedExtent(tile: Int): Int = maxOf(tile + 1, SMALL_PACKED_EXTENT)

    @Test
    fun `a blocked product agrees with the reference in every transpose mode`() = withProductKernels { engine ->
        val rng = Random(20261014)
        for ((m, n, k) in blockedShapes(engine)) {
            // A shape that is not packed would check the panel path instead.
            assertTrue(engine.packsProduct(m, n, k), "${engine.name} ${m}x${n}x$k is not packed")
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    checkProduct(engine, m, n, k, transposeA, transposeB, 0.875, -0.25, rng)
                }
            }
        }
    }

    // Beta reaches the first depth block and nothing after it, so a later block reading the destination
    // again would bring the poison back.
    @Test
    fun `a zero beta overwrites a poisoned destination over several depth blocks`() = withProductKernels { engine ->
        val rng = Random(20261015)
        val k = PRODUCT_BLOCK_DEPTH * 2 + 9
        val m = packedExtent(engine.tileRows)
        val n = packedExtent(engine.tileColumns)
        val a = randomMatrix(m, k, rng)
        val b = randomMatrix(k, n, rng)
        val expected = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })
        ReferenceBlas.gemm(0.875, a, false, b, false, 0.0, expected)
        val c = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })

        engine.gemm(0.875, a, false, b, false, 0.0, c)

        assertClose(expected.values, c.values, "${engine.name} poisoned destination", tolerance = 1e-10)
    }

    // A depth that fits one block and a depth that does not are different traversals of the same arithmetic.
    @Test
    fun `beta reaches the destination once whatever the depth was cut into`() = withProductKernels { engine ->
        val rng = Random(20261016)
        val m = packedExtent(engine.tileRows)
        val n = packedExtent(engine.tileColumns)
        for (k in intArrayOf(PRODUCT_BLOCK_DEPTH - 1, PRODUCT_BLOCK_DEPTH, PRODUCT_BLOCK_DEPTH + 1)) {
            checkProduct(engine, m, n, k, transposeA = false, transposeB = false, 0.5, 2.0, rng)
        }
    }

    // The lengths the workspace holds are worked out from the resolved tile geometry rather than written
    // down, since a packing panel covers whole tiles of the block it serves.
    @Test
    fun `a blocked product stages an operand that shares its destination`() = withProductKernels { engine ->
        val rng = Random(20261017)
        // A square order that is a whole number of tiles on both axes and large enough to be packed.
        val order = lowestCommonTileMultiple(engine.tileRows, engine.tileColumns, atLeast = 40)
        val workspace = Scratch()
        val start = randomMatrix(order, order, rng)
        val expected = DenseMatrix(order, order, start.values.copyOf())
        ReferenceBlas.gemm(
            0.875,
            DenseMatrix(order, order, start.values.copyOf()),
            false,
            DenseMatrix(order, order, start.values.copyOf()),
            true,
            -0.25,
            expected,
        )
        val aliased = DenseMatrix(order, order, start.values.copyOf())

        engine.gemm(0.875, aliased, false, aliased, true, -0.25, aliased, scratch = workspace)

        assertClose(expected.values, aliased.values, "${engine.name} aliased product", tolerance = 1e-10)
        val staged = order * order
        // The panels are borrowed at the capacity the scheduling rounds their extents up to, which is
        // what lets a schedule whose windows shrink reuse one buffer for several of them.
        val leftPanel = scratchCapacity(productBlockRows(engine, order) * productBlockDepth(order))
        val rightPanel = scratchCapacity(productBlockColumns(engine, order) * productBlockDepth(order))
        // One shared operand snapshot and a packing panel for each side. At this order the panels
        // cover the whole of both operands, so on a machine whose blocks are no smaller than the order
        // every one of these lengths is the same.
        assertEquals(
            expectedLoans(staged, listOf(staged, leftPanel, rightPanel)),
            workspace.available(staged),
            "${engine.name} did not share the operand snapshot and pack both panels from the workspace",
        )

        val separate = Scratch()
        val other = randomMatrix(order, order, rng)
        engine.gemm(0.875, other, false, aliased, true, -0.25, expected, scratch = separate)
        assertEquals(
            expectedLoans(staged, listOf(leftPanel, rightPanel)),
            separate.available(staged),
            "${engine.name} staged an operand that shared nothing with its destination",
        )
    }

    /** A zero multiplier scales the destination and reads no operand, whatever stands in them. */
    @Test
    fun `a zero alpha scales the destination without reading an operand`() = withProductKernels { engine ->
        val m = 4 * engine.tileRows
        val n = 4 * engine.tileColumns
        val k = PRODUCT_BLOCK_DEPTH
        val a = DenseMatrix(m, k, DoubleArray(m * k) { Double.NaN })
        val b = DenseMatrix(k, n, DoubleArray(k * n) { Double.POSITIVE_INFINITY })
        val c = DenseMatrix(m, n, DoubleArray(m * n) { 2.0 })

        engine.gemm(0.0, a, false, b, false, 0.5, c)

        assertTrue(c.values.all { it == 1.0 }, "${engine.name} read an operand a zero alpha excludes")
    }

    // Scratch is sized from the operands, which say nothing about whether there is an output to write, so a
    // wide left operand against a right one of no columns would otherwise retain a column for nothing, and an
    // empty extent past several blocks would otherwise walk them.
    @Test
    fun `an empty destination borrows nothing on any product entry point`() = withProductKernels { engine ->
        val rng = Random(20261025)
        val workspace = Scratch()
        val a = randomMatrix(128, 1, rng)
        val b = DenseMatrix(1, 0, DoubleArray(0))
        val c = DenseMatrix(128, 0, DoubleArray(0))
        val wide = 2 * PRODUCT_BLOCK_DEPTH + 1
        val none = DenseMatrix(0, wide, DoubleArray(0))
        val other = DenseMatrix(wide, 0, DoubleArray(0))
        val empty = DenseMatrix(0, 0, DoubleArray(0))

        engine.gemm(1.0, a, false, b, false, 1.0, c, scratch = workspace)
        engine.gemm(0.875, none, false, other, false, -0.25, empty, scratch = workspace)

        assertEquals(0, workspace.idleLengths(), "${engine.name} borrowed scratch for a product with no output")
    }

    /** The operands a product reads are still what they were when it returns. */
    @Test
    fun `a blocked product leaves its operands alone`() = withProductKernels { engine ->
        val rng = Random(20261018)
        val a = randomMatrix(37, 41, rng)
        val b = randomMatrix(41, 29, rng)
        val beforeA = a.values.copyOf()
        val beforeB = b.values.copyOf()

        engine.gemm(0.875, a, false, b, false, -0.25, randomMatrix(37, 29, rng))

        assertContentEquals(beforeA, a.values, "${engine.name} wrote into its left operand")
        assertContentEquals(beforeB, b.values, "${engine.name} wrote into its right operand")
    }

    /** A repeated product over one shape reuses the packing panels rather than allocating new ones. */
    @Test
    fun `a repeated blocked product reuses its packing scratch`() {
        val rng = Random(20261019)
        val workspace = Scratch()
        val engine = ScalarEngine
        val a = randomMatrix(37, 41, rng)
        val b = randomMatrix(41, 29, rng)
        val c = randomMatrix(37, 29, rng)
        // Counted as buffers rather than as lengths, because the two panels are borrowed at rounded
        // capacities and a shape whose panels round to the same one leaves a single length behind.
        val left = scratchCapacity(productBlockRows(engine, 37) * productBlockDepth(41))
        val right = scratchCapacity(productBlockColumns(engine, 29) * productBlockDepth(41))

        engine.gemm(0.875, a, false, b, false, -0.25, c, scratch = workspace)
        val afterOne = workspace.available(left) + if (right == left) 0 else workspace.available(right)
        repeat(3) { engine.gemm(0.875, a, false, b, false, -0.25, c, scratch = workspace) }

        assertEquals(2, afterOne, "a blocked product did not borrow a panel for each operand")
        assertEquals(
            afterOne,
            workspace.available(left) + if (right == left) 0 else workspace.available(right),
            "a repeated product kept asking for new panels",
        )
    }

    @Test
    fun `gemmt agrees with the oracle on both routes in every orientation`() = withEngines { blas ->
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
    fun `the rank updates agree with the oracle on both routes in every orientation`() = withEngines { blas ->
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
    fun `a selected triangle product leaves the opposite triangle exactly as it found it`() = withEngines { blas ->
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
    fun `beta reaches a selected entry once however the depth is cut`() = withProductKernels { products ->
        val order = blockedOrder(products)
        val depth = maxOf(blockedDepth(products, order), PRODUCT_BLOCK_DEPTH + 7)
        assertTrue(depth > PRODUCT_BLOCK_DEPTH, "the fixture does not cut the depth")
        assertTrue(
            products.packsProduct(order, order, depth),
            "${products.name}: ${order}x$depth does not take the blocked route",
        )
        for (lower in booleanArrayOf(false, true)) {
            val recorder = RecordingProducts(products)
            val blas = engineWithKernels(ScalarEngine, panelsFor(products), recorder)
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
    fun `a packed rank update over its own destination agrees with the oracle`() = withProductKernels { products ->
        val order = packedSquareOrder(products)
        assertTrue(
            products.packsProduct(order, order, order),
            "${products.name}: a square of $order does not take the blocked route",
        )
        val blas = engineWithKernels(ScalarEngine, panelsFor(products), products)
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

    // Guard values cover entries outside the window, including gaps between columns.
    @Test
    fun `a product window writes only inside the offsets and the triangle it was given`() =
        withProductKernels { products ->
            for (selected in listOf(OutputTriangle.Full, OutputTriangle.Lower, OutputTriangle.Upper)) {
                for (packed in booleanArrayOf(false, true)) {
                    for (transposeA in booleanArrayOf(false, true)) {
                        assertWindowed(products, selected, packed, transposeA, transposeB = !transposeA)
                    }
                }
            }
        }

    @Test
    fun `a product window with no depth scales only its selected entries`() = withProductKernels { products ->
        val order = 5
        val leading = order + 2
        val stored = 2.0
        for (selected in listOf(OutputTriangle.Full, OutputTriangle.Lower, OutputTriangle.Upper)) {
            val destination = DoubleArray(C_ORIGIN + order * leading) { GUARD }
            for (column in 0 until order) {
                for (row in 0 until order) {
                    destination[C_ORIGIN + row + column * leading] = stored
                }
            }

            productWindow(
                engineWithKernels(products = products, panels = panelsFor(products)),
                ALPHA, DoubleArray(0), 0, 1, false, DoubleArray(0), 0, 1, false,
                BETA, destination, C_ORIGIN, leading, order, order, 0, selected, null,
            )

            for (column in 0 until order) {
                for (row in 0 until leading) {
                    val expected = when {
                        row >= order -> GUARD
                        outsideTriangle(row, 1, column, 1, selected) -> stored
                        else -> stored * BETA
                    }
                    assertEquals(
                        expected,
                        destination[C_ORIGIN + row + column * leading],
                        "$selected at ($row, $column)",
                    )
                }
            }
        }
    }

    @Suppress("LongParameterList") // the backend and the four facts that pick out one case of the sweep
    private fun assertWindowed(
        products: EngineKernels,
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
            products.packsProduct(order, order, depth),
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
            engineWithKernels(
                products = products,
                panels = panelsFor(products),
            ),
            ALPHA, left, A_ORIGIN, a.rows + 3, transposeA, right, B_ORIGIN, b.rows + 5, transposeB,
            BETA, destination, C_ORIGIN,
            order + 7, order, order, depth, selected, Scratch(),
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

    /** A rank update whose operand is its own destination reads the operand it was given, not the result. */
    @Test
    fun `a rank update over its own destination agrees with the oracle`() = withEngines { blas ->
        val order = 2 * SCALAR_PRODUCT_TILE + 1
        val start = randomMatrix(order, order, rng)
        for (lower in booleanArrayOf(false, true)) {
            val expected = start.copyOf()
            ReferenceBlas.syrk(0.875, start.copyOf(), false, -0.25, expected, lower)
            val aliased = start.copyOf()

            blas.syrk(0.875, aliased, false, -0.25, aliased, lower)

            assertClose(expected.values, aliased.values, "syrk into its own operand lower=$lower", TOLERANCE)
        }
    }

    // Run under a recorder, so the claim is checked against the blocks the schedule produced.
    @Test
    fun `a selected triangle product writes only the blocks inside its triangle`() = withProductKernels { products ->
        val panels = panelsFor(products)
        val order = blockedOrder(products)
        val depth = blockedDepth(products, order)
        for (lower in booleanArrayOf(false, true)) {
            assertTriangleProductStaysInTriangle(products, panels, order, depth, lower)
            assertTriangleProductStaysInTriangle(products, panels, directOrder, directDepth, lower)
            assertTriangleProductStaysInTriangle(
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

    private fun koblasPanels(): EngineKernels = BuiltinEngines.simd?.panelProvider ?: ScalarEngine

    /** The panel backend that belongs beside [products], so a composition under test is a real one. */
    private fun panelsFor(products: EngineKernels): EngineKernels =
        if (products === ScalarEngine) ScalarEngine else koblasPanels()

    /** The blocked shape and the direct one, so every conformance check covers both routes. */
    private fun shapes(): List<Pair<Int, Int>> {
        val products = BuiltinEngines.simd?.productProvider ?: ScalarEngine
        val order = blockedOrder(products)
        return listOf(order to blockedDepth(products, order), directOrder to directDepth)
    }

    /** How many of [lengths] came out at [length], which is what a workspace holds idle at that length. */
    private fun expectedLoans(length: Int, lengths: List<Int>): Int = lengths.count { it == length }

    /** The smallest common multiple of both tile dimensions at or above [atLeast]. */
    private fun lowestCommonTileMultiple(rows: Int, columns: Int, atLeast: Int): Int {
        var step = rows
        while (step % columns != 0) step += rows
        var order = step
        while (order < atLeast) order += step
        return order
    }

    @Suppress("LongParameterList") // the product, both transpose flags, both multipliers and the source
    private fun checkProduct(
        engine: KoblasEngine,
        m: Int,
        n: Int,
        k: Int,
        transposeA: Boolean,
        transposeB: Boolean,
        alpha: Double,
        beta: Double,
        rng: Random,
    ) {
        val a = randomMatrix(if (transposeA) k else m, if (transposeA) m else k, rng)
        val b = randomMatrix(if (transposeB) n else k, if (transposeB) k else n, rng)
        val start = randomMatrix(m, n, rng)
        val expected = DenseMatrix(m, n, start.values.copyOf())
        ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, expected)
        val c = DenseMatrix(m, n, start.values.copyOf())

        engine.gemm(alpha, a, transposeA, b, transposeB, beta, c)

        assertClose(
            expected.values,
            c.values,
            "${engine.name} ${m}x${n}x$k tA=$transposeA tB=$transposeB",
            tolerance = 1e-10,
        )
    }

    /**
     * A product too small or too thin to hide a copy is not packed, and a large square one is.
     *
     * The rule is the backend's, so what is asserted is its shape rather than its constants: an extent below
     * one tile cannot fill the tiles it would pack into, and a matrix-vector shaped call has no arithmetic to
     * amortise a copy over however large its other extent is.
     */
    @Test
    fun `packing is declined where there is nothing to amortise it over`() = withProductKernels { kernels ->
        assertFalse(kernels.packsProduct(1, 1, 1), "${kernels.name} packed a single product")
        assertFalse(kernels.packsProduct(4096, 1, 4096), "${kernels.name} packed a matrix-vector product")
        assertFalse(kernels.packsProduct(1, 4096, 4096), "${kernels.name} packed a vector-matrix product")
        assertTrue(kernels.packsProduct(256, 256, 256), "${kernels.name} declined a large square product")
    }

    // A fixture chosen on one machine can quietly become a test of the unpacked route on another, so the rule
    // is asked directly at tile widths this host may lack: both extents must reach a whole tile, and the work
    // must reach the threshold.
    @Test
    fun `the packing rule needs a whole tile on each axis and enough work at every tile geometry`() {
        for (tileRows in intArrayOf(2, 4, 8, 16, 32)) {
            val tileColumns = 4
            val depth = (PACKED_PRODUCT_MINIMUM_WORK / (tileRows * tileColumns)).toInt() + 1
            val context = "a tile of $tileRows by $tileColumns"

            assertTrue(packsProductByWork(tileRows, tileColumns, depth, tileRows, tileColumns), context)
            assertFalse(packsProductByWork(tileRows - 1, 4096, depth, tileRows, tileColumns), "$context rows")
            assertFalse(packsProductByWork(4096, tileColumns - 1, depth, tileRows, tileColumns), "$context columns")
            assertFalse(packsProductByWork(tileRows, tileColumns, depth - 2, tileRows, tileColumns), "$context work")
        }
    }

    private companion object {
        /**
         * The smallest extent these shapes use on an axis they are not blocking: over the packing threshold
         * at every tile geometry, and a multiple of none of them so the last tile is always a remainder.
         */
        const val SMALL_PACKED_EXTENT = 25

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
