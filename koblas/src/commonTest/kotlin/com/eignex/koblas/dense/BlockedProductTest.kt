package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.vendor.RouteKind
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The matrix product where it is large enough to be packed and blocked, on every built-in engine.
 *
 * The shapes here are chosen to reach what block scheduling has that a small product does not: more than one
 * cache block on each axis, a depth cut into several blocks so that beta has somewhere to go wrong, and
 * extents that fill no tile exactly. Every answer is compared with [ReferenceBlas], which shares no code
 * with any of it.
 */
class BlockedProductTest {
    private val engines: List<KoblasEngine> get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)

    /**
     * Shapes that cross one block boundary each, kept small on the other two axes so the oracle is cheap.
     *
     * The last is neither blocked on any axis nor a multiple of any tile, which is the edge case the three
     * before it do not have.
     *
     * The small extents follow the tile this machine resolved, because whether a product is packed at all
     * depends on it: a shape that fills more than one tile at four rows fills none at sixteen, and the test
     * would then quietly be checking the unpacked route. `packedExtent` is a tile and a remainder, and never
     * so few that the product falls under the packing threshold.
     */
    private fun blockedShapes(tile: DenseProductKernels): List<Triple<Int, Int, Int>> {
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

    /**
     * That the shapes above are packed at every tile geometry, not only the one this machine resolved.
     *
     * Whether a product is packed at all depends on the tile: four rows and sixteen rows disagree about
     * whether a shape fills one, so a fixture chosen on one machine can quietly become a test of the
     * unpacked route on another. That is not hypothetical, and it is not something a machine with four lanes
     * can find by running anything, so the rule is asked directly at the widths this machine cannot resolve.
     */
    @Test
    fun `the blocked shapes are packed at every tile geometry`() {
        for (tileRows in intArrayOf(2, 4, 8, 16, 32)) {
            val tile = TileGeometry(tileRows, 4)
            for ((m, n, k) in blockedShapes(tile)) {
                assertTrue(
                    tile.packsProduct(m, n, k),
                    "${m}x${n}x$k is not packed by a tile of $tileRows rows, so it would test the other route",
                )
            }
        }
    }

    /** A tile of a stated shape that answers the shipped packing rule and nothing else. */
    private class TileGeometry(override val tileRows: Int, override val tileColumns: Int) : DenseProductKernels {
        override val name: String get() = "geometry(${tileRows}x$tileColumns)"

        override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> = listOf(name)

        override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean =
            packsProductByWork(rows, columns, depth, tileRows, tileColumns)

        @Suppress("LongParameterList") // the product block contract
        override fun productBlock(
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
        ): Unit = throw UnsupportedOperationException("this stands in for a geometry, not for arithmetic")
    }

    @Test
    fun `a blocked product agrees with the reference across every transpose pair`() {
        val rng = Random(20261014)
        for (engine in engines) {
            for ((m, n, k) in blockedShapes(engine.productKernels)) {
                for (transposeA in booleanArrayOf(false, true)) {
                    for (transposeB in booleanArrayOf(false, true)) {
                        checkProduct(engine, m, n, k, transposeA, transposeB, 0.875, -0.25, rng)
                    }
                }
            }
        }
    }

    /**
     * That these shapes really are the packed route, so the test above is not quietly checking the panel one.
     *
     * Asked of the route rather than assumed, and the route is what the call itself decides from. A shape
     * whose rows fill whole tiles reaches one body and is direct; one that leaves a remainder reaches the
     * body that remainder goes to as well, and is published as the composition it is.
     */
    @Test
    fun `the blocked shapes reach the product tiles and a small one does not`() {
        for (engine in engines) {
            for ((m, n, k) in blockedShapes(engine.productKernels)) {
                val route = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(m, n, depth = k))
                assertTrue(
                    route.components.any { it.endsWith("/product-block") },
                    "${engine.name} ${m}x${n}x$k named ${route.components}",
                )
            }
            val tile = engine.productKernels
            val whole = engine.denseRouteOf(
                DenseMatrixOperation.Gemm,
                DenseCall(tile.tileRows * 8, tile.tileColumns * 8, depth = 64),
            )
            assertEquals(RouteKind.Direct, whole.kind, "${engine.name} $whole")
            val small = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(5, 4, depth = 6))
            assertTrue(
                small.components.none { it.endsWith("/product-block") },
                "${engine.name} packed a product with nothing to amortise it over: ${small.components}",
            )
        }
    }

    /**
     * A destination arriving as NaN, overwritten rather than multiplied, across several depth blocks.
     *
     * The depth is what makes it more than the tile's own rule: beta reaches the first depth block and
     * nothing after it, so a later block reading the destination again would bring the poison back.
     */
    @Test
    fun `a zero beta overwrites a poisoned destination over several depth blocks`() {
        val rng = Random(20261015)
        val k = PRODUCT_BLOCK_DEPTH * 2 + 9
        for (engine in engines) {
            val m = packedExtent(engine.productKernels.tileRows)
            val n = packedExtent(engine.productKernels.tileColumns)
            val a = randomMatrix(m, k, rng)
            val b = randomMatrix(k, n, rng)
            val expected = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })
            ReferenceBlas.gemm(0.875, a, false, b, false, 0.0, expected)
            val c = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })

            engine.gemm(0.875, a, false, b, false, 0.0, c)

            assertClose(expected.values, c.values, "${engine.name} poisoned destination", tolerance = 1e-10)
        }
    }

    /**
     * That beta is applied once across depth blocks, which the same product cut into one block cannot show.
     *
     * A depth that fits one block and a depth that does not are different traversals of the same arithmetic,
     * and the reference answers both.
     */
    @Test
    fun `beta reaches the destination once whatever the depth was cut into`() {
        val rng = Random(20261016)
        for (engine in engines) {
            val m = packedExtent(engine.productKernels.tileRows)
            val n = packedExtent(engine.productKernels.tileColumns)
            for (k in intArrayOf(PRODUCT_BLOCK_DEPTH - 1, PRODUCT_BLOCK_DEPTH, PRODUCT_BLOCK_DEPTH + 1)) {
                checkProduct(engine, m, n, k, transposeA = false, transposeB = false, 0.5, 2.0, rng)
            }
        }
    }

    /**
     * A product whose operands are its own destination, staged so the packing reads what the caller gave.
     *
     * The lengths the workspace ends up holding follow from the tile geometry this machine resolved, since a
     * packing panel covers whole tiles of the block it serves. They are worked out here from that geometry
     * rather than written down, because a tile eight rows deep and one sixteen rows deep round the same
     * order to different panels and neither is this test's business.
     */
    @Test
    fun `a blocked product stages an operand that shares its destination`() {
        val rng = Random(20261017)
        for (engine in engines) {
            val tile = engine.productKernels
            // A square order that is a whole number of tiles on both axes and large enough to be packed.
            val order = lowestCommonTileMultiple(tile.tileRows, tile.tileColumns, atLeast = 40)
            val workspace = Workspace()
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

            engine.gemm(0.875, aliased, false, aliased, true, -0.25, aliased, workspace)

            assertClose(expected.values, aliased.values, "${engine.name} aliased product", tolerance = 1e-10)
            val staged = order * order
            // The panels are borrowed at the capacity the scheduling rounds their extents up to, which is
            // what lets a schedule whose windows shrink reuse one buffer for several of them.
            val leftPanel = scratchCapacity(productBlockRows(tile, order) * productBlockDepth(order))
            val rightPanel = scratchCapacity(productBlockColumns(tile, order) * productBlockDepth(order))
            // One staged copy of each aliased operand, and a packing panel for each side. At this order the
            // panels cover the whole of both operands, so on a machine whose blocks are no smaller than the
            // order every one of these lengths is the same.
            assertEquals(
                expectedLoans(staged, listOf(staged, staged, leftPanel, rightPanel)),
                workspace.available(staged),
                "${engine.name} did not stage both aliased operands and pack both panels from the workspace",
            )

            val separate = Workspace()
            engine.gemm(0.875, randomMatrix(order, order, rng), false, aliased, true, -0.25, expected, separate)
            assertEquals(
                expectedLoans(staged, listOf(leftPanel, rightPanel)),
                separate.available(staged),
                "${engine.name} staged an operand that shared nothing with its destination",
            )
        }
    }

    /**
     * That alpha multiplies an accumulated sum and never an entry of an operand, on both routes.
     *
     * An infinite alpha against a zero entry is what tells the two apart: scaling the entry gives a NaN that
     * the sum then carries, and scaling the sum gives the infinity the definition asks for. Every dot here
     * is exactly one, so the answer the contract states is one infinity per entry.
     */
    @Test
    fun `alpha multiplies a sum of products rather than an entry of an operand`() {
        for (engine in engines) {
            val tile = engine.productKernels
            val direct = DenseMatrix(1, 1, DoubleArray(1) { Double.NaN })
            val row = DenseMatrix(1, 2, doubleArrayOf(1.0, 1.0))
            val column = DenseMatrix(2, 1, doubleArrayOf(0.0, 1.0))
            engine.gemm(Double.POSITIVE_INFINITY, row, false, column, false, 0.0, direct)
            assertEquals(Double.POSITIVE_INFINITY, direct.values[0], "${engine.name} unpacked product")

            val transposed = DenseMatrix(1, 1, DoubleArray(1) { Double.NaN })
            val stored = DenseMatrix(2, 1, doubleArrayOf(1.0, 1.0))
            engine.gemm(Double.POSITIVE_INFINITY, stored, true, column, false, 0.0, transposed)
            assertEquals(Double.POSITIVE_INFINITY, transposed.values[0], "${engine.name} transposed product")

            val m = 4 * tile.tileRows
            val n = 4 * tile.tileColumns
            val k = PRODUCT_BLOCK_DEPTH
            val a = DenseMatrix(m, k, DoubleArray(m * k) { 1.0 })
            val b = DenseMatrix(k, n, DoubleArray(k * n) { if (it % k == k - 1) 1.0 else 0.0 })
            val c = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })
            assertTrue(tile.packsProduct(m, n, k), "${engine.name} did not pack ${m}x${n}x$k")

            engine.gemm(Double.POSITIVE_INFINITY, a, false, b, false, 0.0, c)

            assertTrue(
                c.values.all { it == Double.POSITIVE_INFINITY },
                "${engine.name} packed product produced ${c.values.distinct().take(3)}",
            )
        }
    }

    /**
     * The declared limit of that rule: which partition of the depth is summed before alpha is the schedule's.
     *
     * One depth block sums the whole shared dimension and multiplies once; two blocks multiply each of their
     * sums and add the results, which for an infinite alpha against a block that sums to zero is a NaN. That
     * is a consequence of blocking rather than an accident, and [com.eignex.koblas.dense.DenseBlas.gemm] says
     * so; what is pinned here is that the behaviour is the stated one and not something else.
     */
    @Test
    fun `an infinite alpha follows the depth partition the schedule chose`() {
        for (engine in engines) {
            val tile = engine.productKernels
            val m = 4 * tile.tileRows
            val n = 4 * tile.tileColumns
            for ((k, expected) in listOf(PRODUCT_BLOCK_DEPTH to false, PRODUCT_BLOCK_DEPTH + 1 to true)) {
                val a = DenseMatrix(m, k, DoubleArray(m * k) { 1.0 })
                val b = DenseMatrix(k, n, DoubleArray(k * n) { if (it % k == k - 1) 1.0 else 0.0 })
                val c = DenseMatrix(m, n, DoubleArray(m * n))

                engine.gemm(Double.POSITIVE_INFINITY, a, false, b, false, 0.0, c)

                assertEquals(
                    expected,
                    c.values.all { it.isNaN() },
                    "${engine.name} at depth $k produced ${c.values.first()}",
                )
            }
        }
    }

    /** A zero multiplier scales the destination and reads no operand, whatever stands in them. */
    @Test
    fun `a zero alpha scales the destination without reading an operand`() {
        for (engine in engines) {
            val tile = engine.productKernels
            val m = 4 * tile.tileRows
            val n = 4 * tile.tileColumns
            val k = PRODUCT_BLOCK_DEPTH
            val a = DenseMatrix(m, k, DoubleArray(m * k) { Double.NaN })
            val b = DenseMatrix(k, n, DoubleArray(k * n) { Double.POSITIVE_INFINITY })
            val c = DenseMatrix(m, n, DoubleArray(m * n) { 2.0 })

            engine.gemm(0.0, a, false, b, false, 0.5, c)

            assertTrue(c.values.all { it == 1.0 }, "${engine.name} read an operand a zero alpha excludes")
        }
    }

    /**
     * Extents with nothing in them, including one large enough that walking it would be the whole cost.
     *
     * A matrix with no rows and a hundred thousand columns has no entries, so the product has nothing to
     * pack, nothing to compute and no group to step through; settling that before any of it is what keeps an
     * empty shape cheap rather than proportional to the dimension it does not have.
     */
    @Test
    fun `a product over an empty extent does nothing and costs nothing`() {
        val wide = 1_000_000
        for (engine in engines) {
            val none = DenseMatrix(0, wide, DoubleArray(0))
            val other = DenseMatrix(wide, 0, DoubleArray(0))
            val empty = DenseMatrix(0, 0, DoubleArray(0))

            engine.gemm(0.875, none, false, other, false, -0.25, empty)

            val left = engine.packLeft(none, transpose = false)
            val right = engine.packRight(other, transpose = false)
            assertEquals(0, left.layout.storageSize, "${engine.name} left panel of an empty matrix")
            assertEquals(0, right.layout.storageSize, "${engine.name} right panel of an empty matrix")

            val destination = DenseMatrix(0, 0, DoubleArray(0))
            engine.gemm(0.875, left, right, -0.25, destination)
        }
    }

    /**
     * A destination with no entries takes no scratch on any of the product entry points.
     *
     * The scratch a product borrows is sized from its operands, which say nothing about whether there is an
     * output to write. A left operand of a hundred and twenty-eight rows against a right operand of no
     * columns would otherwise retain a column of that length for a result with no entries in it, and a
     * larger operand a correspondingly larger one.
     */
    @Test
    fun `an empty destination borrows nothing on any product entry point`() {
        val rng = Random(20261025)
        for (engine in engines) {
            val workspace = Workspace()
            val a = randomMatrix(128, 1, rng)
            val b = DenseMatrix(1, 0, DoubleArray(0))
            val c = DenseMatrix(128, 0, DoubleArray(0))
            val left = engine.packLeft(a, transpose = false)
            val right = engine.packRight(b, transpose = false)

            engine.gemm(1.0, a, false, b, false, 1.0, c, workspace)
            engine.gemm(1.0, left, right, 1.0, c)
            engine.gemm(1.0, left, b, false, 1.0, c, workspace)
            engine.gemm(1.0, a, false, right, 1.0, c, workspace)

            assertEquals(
                0,
                workspace.idleLengths(),
                "${engine.name} borrowed scratch for a product with no output",
            )
        }
    }

    /** A depth of nothing scales the destination and reads neither operand, at any other extent. */
    @Test
    fun `a product with no shared dimension scales its destination`() {
        for (engine in engines) {
            val rows = 100_000
            val a = DenseMatrix(rows, 0, DoubleArray(0))
            val b = DenseMatrix(0, 1, DoubleArray(0))
            val c = DenseMatrix(rows, 1, DoubleArray(rows) { 4.0 })

            engine.gemm(0.875, a, false, b, false, 0.25, c)

            assertTrue(c.values.all { it == 1.0 }, "${engine.name} did not scale a destination with no depth")
        }
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

    /** The operands a product reads are still what they were when it returns. */
    @Test
    fun `a blocked product leaves its operands alone`() {
        val rng = Random(20261018)
        for (engine in engines) {
            val a = randomMatrix(37, 41, rng)
            val b = randomMatrix(41, 29, rng)
            val beforeA = a.values.copyOf()
            val beforeB = b.values.copyOf()

            engine.gemm(0.875, a, false, b, false, -0.25, randomMatrix(37, 29, rng))

            assertEquals(beforeA.toList(), a.values.toList(), "${engine.name} wrote into its left operand")
            assertEquals(beforeB.toList(), b.values.toList(), "${engine.name} wrote into its right operand")
        }
    }

    /** A repeated product over one shape reuses the packing panels rather than allocating new ones. */
    @Test
    fun `a repeated blocked product reuses its packing scratch`() {
        val rng = Random(20261019)
        val workspace = Workspace()
        val engine = koblasEngineUnderTest()
        val tile = engine.productKernels
        val a = randomMatrix(37, 41, rng)
        val b = randomMatrix(41, 29, rng)
        val c = randomMatrix(37, 29, rng)
        // Counted as buffers rather than as lengths, because the two panels are borrowed at rounded
        // capacities and a shape whose panels round to the same one leaves a single length behind.
        val left = scratchCapacity(productBlockRows(tile, 37) * productBlockDepth(41))
        val right = scratchCapacity(productBlockColumns(tile, 29) * productBlockDepth(41))

        engine.gemm(0.875, a, false, b, false, -0.25, c, workspace)
        val afterOne = workspace.available(left) + if (right == left) 0 else workspace.available(right)
        repeat(3) { engine.gemm(0.875, a, false, b, false, -0.25, c, workspace) }

        assertEquals(2, afterOne, "a blocked product did not borrow a panel for each operand")
        assertEquals(
            afterOne,
            workspace.available(left) + if (right == left) 0 else workspace.available(right),
            "a repeated product kept asking for new panels",
        )
    }

    /**
     * The three retained forms against the reference, over every transpose pair.
     *
     * A retained panel is packed from `op(A)`, so the transpose is spent at packing time and the product
     * itself has none; what the pair still changes is what the panel has to contain. The mixed forms pack
     * the other operand for the call, which is where the transpose flag survives.
     */
    @Test
    fun `a product over retained panels agrees with the reference`() {
        val rng = Random(20261021)
        for (engine in engines) {
            val rows = packedExtent(engine.productKernels.tileRows)
            val columns = packedExtent(engine.productKernels.tileColumns)
            for ((m, n, k) in listOf(
                Triple(37, 29, 41),
                Triple(rows, columns, PRODUCT_BLOCK_DEPTH + 9),
                Triple(4, 4, 4),
            )) {
                for (transposeA in booleanArrayOf(false, true)) {
                    for (transposeB in booleanArrayOf(false, true)) {
                        checkRetainedProduct(engine, m, n, k, transposeA, transposeB, rng)
                    }
                }
            }
        }
    }

    /** A retained product with nothing to compute scales its destination and reads no panel. */
    @Test
    fun `a retained product with no work scales its destination`() {
        val rng = Random(20261022)
        for (engine in engines) {
            val left = engine.packLeft(randomMatrix(6, 0, rng), transpose = false)
            val right = engine.packRight(randomMatrix(0, 5, rng), transpose = false)
            val c = DenseMatrix(6, 5, DoubleArray(30) { 4.0 })

            engine.gemm(2.0, left, right, 0.5, c)
            assertTrue(c.values.all { it == 2.0 }, "${engine.name} did not scale an empty product")

            engine.gemm(0.0, left, right, 0.0, c)
            assertTrue(c.values.all { it == 0.0 }, "${engine.name} did not overwrite for a zero multiplier")
        }
    }

    /**
     * A mixed retained product whose dense operand is also its destination.
     *
     * The packing reads that operand block by block while the destination is being written, so without
     * staging a later block would pack values the product itself had just produced.
     */
    @Test
    fun `a mixed retained product stages a dense operand that shares its destination`() {
        val rng = Random(20261023)
        for (engine in engines) {
            val order = lowestCommonTileMultiple(
                engine.productKernels.tileRows,
                engine.productKernels.tileColumns,
                atLeast = 40,
            )
            val start = randomMatrix(order, order, rng)
            val retained = engine.packLeft(randomMatrix(order, order, rng), transpose = false)
            val left = DenseMatrix(order, order, DoubleArray(order * order) { retained[it % order, it / order] })
            val expected = DenseMatrix(order, order, start.values.copyOf())
            ReferenceBlas.gemm(
                0.875,
                left,
                false,
                DenseMatrix(order, order, start.values.copyOf()),
                false,
                -0.25,
                expected,
            )
            val aliased = DenseMatrix(order, order, start.values.copyOf())

            engine.gemm(0.875, retained, aliased, false, -0.25, aliased, Workspace())

            assertClose(expected.values, aliased.values, "${engine.name} mixed aliased", tolerance = 1e-10)
        }
    }

    @Suppress("LongParameterList") // the product, both transpose flags and the source
    private fun checkRetainedProduct(
        engine: KoblasEngine,
        m: Int,
        n: Int,
        k: Int,
        transposeA: Boolean,
        transposeB: Boolean,
        rng: Random,
    ) {
        val a = randomMatrix(if (transposeA) k else m, if (transposeA) m else k, rng)
        val b = randomMatrix(if (transposeB) n else k, if (transposeB) k else n, rng)
        val start = randomMatrix(m, n, rng)
        val expected = DenseMatrix(m, n, start.values.copyOf())
        ReferenceBlas.gemm(0.875, a, transposeA, b, transposeB, -0.25, expected)
        val left = engine.packLeft(a, transposeA)
        val right = engine.packRight(b, transposeB)
        val context = "${engine.name} retained ${m}x${n}x$k tA=$transposeA tB=$transposeB"

        val both = DenseMatrix(m, n, start.values.copyOf())
        engine.gemm(0.875, left, right, -0.25, both)
        assertClose(expected.values, both.values, "$context both retained", tolerance = 1e-10)

        val leftOnly = DenseMatrix(m, n, start.values.copyOf())
        engine.gemm(0.875, left, b, transposeB, -0.25, leftOnly, Workspace())
        assertClose(expected.values, leftOnly.values, "$context left retained", tolerance = 1e-10)

        val rightOnly = DenseMatrix(m, n, start.values.copyOf())
        engine.gemm(0.875, a, transposeA, right, -0.25, rightOnly, Workspace())
        assertClose(expected.values, rightOnly.values, "$context right retained", tolerance = 1e-10)
    }

    private fun koblasEngineUnderTest(): KoblasEngine = BuiltinEngines.scalar

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

    private companion object {
        /**
         * The smallest extent these shapes use on an axis they are not blocking.
         *
         * Large enough that the product clears the packing threshold at every tile geometry this library
         * resolves, and not a multiple of any of them, so the last tile on that axis is always a remainder.
         */
        const val SMALL_PACKED_EXTENT = 25
    }
}
