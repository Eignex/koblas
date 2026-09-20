package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.Workspace
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory

/**
 * Runs allocation checks for the Vector API dense panels in an uninstrumented JVM.
 *
 * Kover's test instrumentation prevents HotSpot from scalar-replacing Vector API carriers, which turns an
 * allocation-free panel into a coverage artifact, so this runs through the `simdDenseAllocationCheck` Gradle
 * task rather than a test task.
 *
 * Whole operations, not only raw panels. A panel measured on its own can be allocation-free while the caller
 * around it is not, and it is the caller a user writes; the raw panels are here beside them so a regression
 * says which of the two moved.
 */
internal object SimdDenseAllocationCheck {
    private const val ORDER = 512
    private const val COLUMNS = 8
    private const val MAX_BYTES_PER_CALL = 8.0
    private const val WARMUP_ITERATIONS = 20_000
    private const val MEASUREMENT_ITERATIONS = 2_000
    private const val MEASUREMENT_WINDOWS = 8

    /** A square product large enough to be packed and small enough to repeat thousands of times. */
    private const val PRODUCT_ORDER = 48

    /**
     * The other shape: too few rows and columns to block, and a depth that several blocks cover.
     *
     * A floor rather than the extent itself. A tile sixteen rows deep would not pack twelve rows at all, so
     * the caller raises this to something that machine's tile really packs and says so.
     */
    private const val PRODUCT_NARROW = 12
    private const val PRODUCT_DEPTH = 400

    /** One product is the arithmetic of many panels, so it needs fewer repetitions to reach the same state. */
    private const val PRODUCT_WARMUP = 2_000
    private const val PRODUCT_ITERATIONS = 200

    /** A square order the structured probes use, raised where a machine's tile needs more than three of it. */
    private const val STRUCTURED_ORDER = 49

    /**
     * Diagonal blocks the triangular probe spans, past what a workspace retains distinct lengths for.
     *
     * The point of the probe is the scratch a schedule whose windows shrink asks for, and eight or fewer
     * blocks would fit inside the retention bound without the rounding this stage added.
     */
    private const val TRIANGULAR_BLOCKS = 10

    /** One triangular call is the arithmetic of a whole product, so it repeats fewer times still. */
    private const val TRIANGULAR_WARMUP = 600
    private const val TRIANGULAR_ITERATIONS = 40

    /** Diagonal blocks the second triangular probe spans, several octaves of extent above the first. */
    private const val WIDE_TRIANGULAR_BLOCKS = 32
    private const val WIDE_WARMUP = 200
    private const val WIDE_ITERATIONS = 20

    /** Wider than any grouping a backend recommends, so asking with it returns the recommendation itself. */
    private const val UNGROUPED_SIDES = 4_096

    private val allocationBean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    @Volatile
    private var resultSink = 0.0

    @JvmStatic
    fun main(args: Array<String>) {
        val engine = requireNotNull(BuiltinEngines.simd) {
            "dense allocation check requires the Vector API panel candidate"
        }
        val panels = engine.panelKernels
        require(panels === SimdPanelKernels) { "dense allocation check selected ${panels.name}" }
        require(panels.implementationFor(PanelWork.MultiDot, ORDER, COLUMNS) == panels.name) {
            "dense allocation check uses $ORDER rows, which does not reach the vector body"
        }

        val a = DoubleArray(ORDER * ORDER) { 1.0 + (it % 13) * 0.125 }
        val x = DoubleArray(ORDER) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(ORDER)
        val coefficients = DoubleArray(ORDER) { 0.5 + (it % 5) * 0.125 }
        val sums = DoubleArray(ORDER)

        assertAllocationFree("multi-dot panel") {
            panels.multiDot(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, 0.0, y, 0, 1)
            y[0]
        }
        assertAllocationFree("column-update panel") {
            panels.columnUpdate(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, y, 0, 1)
            y[0]
        }
        assertAllocationFree("coupled panel") {
            panels.coupledUpdateDot(0.875, a, 0, ORDER, x, 0, ORDER, COLUMNS, y, 0, coefficients, 0, sums, 0)
            sums[0]
        }
        assertAllocationFree("rank-update panel") {
            panels.rankUpdate(0.875, a, 0, ORDER, x, 0, 1, ORDER, COLUMNS, coefficients, 0, 1)
            a[0]
        }
        checkWholeOperations(engine)
        checkProducts(engine)
        checkStructuredProducts(engine)
        checkTriangularOperations(engine)
    }

    /**
     * The Level 3 routines whose destination or operand is a triangle, as complete calls.
     *
     * These reach paths no rectangular product does: a block straddling the diagonal accumulates into a
     * borrowed tile and merges part of it out, a symmetric operand's diagonal blocks are written into a
     * borrowed square, and a rank-2k update runs the whole schedule twice. All of that is scratch taken
     * from a workspace and handed back, so a loan that escaped or a wrapper that survived shows up here.
     */
    private fun checkStructuredProducts(engine: KoblasEngine) {
        val products = engine.productKernels
        val order = maxOf(products.tileRows * 3 + 1, STRUCTURED_ORDER)
        require(products.packsProduct(order, order, order)) {
            "the structured probe at $order cubed is not packed on this machine"
        }
        val workspace = Workspace()
        val square = order * order
        val a = DenseMatrix.wrap(order, order, DoubleArray(square) { 1.0 + (it % 13) * 0.125 })
        val b = DenseMatrix.wrap(order, order, DoubleArray(square) { 0.5 + (it % 7) * 0.25 })
        val c = DenseMatrix.wrap(order, order, DoubleArray(square))

        for (lower in booleanArrayOf(true, false)) {
            assertAllocationFree("gemmt lower=$lower", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
                engine.gemmt(1e-12, a, false, b, false, -0.25, c, lower, workspace)
                c.values[0]
            }
            assertAllocationFree("syrk lower=$lower", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
                engine.syrk(1e-12, a, false, -0.25, c, lower, workspace)
                c.values[0]
            }
            assertAllocationFree("syr2k lower=$lower", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
                engine.syr2k(1e-12, a, b, false, -0.25, c, lower, workspace)
                c.values[0]
            }
        }
        for (right in booleanArrayOf(false, true)) {
            assertAllocationFree("symm right=$right", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
                engine.symm(1e-12, a, b, -0.25, c, lower = true, right = right, workspace = workspace)
                c.values[0]
            }
        }
    }

    /**
     * The triangular routines, over the two things about them a narrow probe would miss.
     *
     * One is the thin call: a left solve over a single right-hand side accumulates a destination column for
     * every diagonal block, and those columns are all different lengths, so a schedule that borrowed at the
     * exact extent would exhaust what a workspace retains and allocate again on every warmed call. The
     * order here is past eight diagonal blocks for that reason, which neither a short call nor a wide one
     * reaches.
     *
     * The other is the substitution itself. A left call leaves its right-hand sides strided and the backend
     * may gather a block of them; a right call finds them adjacent and gathers nothing; a unit diagonal
     * removes the division from every step; and a side count that is not a whole number of lane blocks ends
     * in a scalar body. Both substitutions run on both sides, so the multiply is checked over already
     * adjacent right-hand sides as well as over gathered ones. Each of those is a different arrangement of
     * the same loop, and each is here.
     */
    private fun checkTriangularOperations(engine: KoblasEngine) {
        checkTriangularOrder(engine, TRIANGULAR_BLOCKS * TRIANGULAR_DIAGONAL_BLOCK)
        // A second order several octaves above the first, over a single right-hand side. What the schedule
        // borrows has to be bounded by its own blocks rather than by the call, and a bound demonstrated at
        // one size would be a property of that size. Fewer repetitions, because one call of this is the
        // arithmetic of many of the other.
        checkThinSolve(engine, WIDE_TRIANGULAR_BLOCKS * TRIANGULAR_DIAGONAL_BLOCK)
    }

    private fun checkThinSolve(engine: KoblasEngine, order: Int) {
        val workspace = Workspace()
        val triangle = dominantTriangle(order)
        val b = DenseMatrix.wrap(order, 1, DoubleArray(order))

        assertAllocationFree("trsm left order=$order sides=1", WIDE_WARMUP, WIDE_ITERATIONS) {
            b.values.fill(1.0)
            engine.trsm(triangle, b, lower = true, workspace = workspace)
            b.values[0]
        }
    }

    private fun dominantTriangle(order: Int): DenseMatrix {
        val triangle = DenseMatrix.wrap(
            order,
            order,
            DoubleArray(order * order) { 0.25 + (it % 11) * 0.0625 },
        )
        for (i in 0 until order) triangle.values[i + i * order] = 4.0
        return triangle
    }

    private fun checkTriangularOrder(engine: KoblasEngine, order: Int) {
        val workspace = Workspace()
        val triangle = dominantTriangle(order)
        val group = engine.triangularKernels.rightHandSideGroup(TRIANGULAR_DIAGONAL_BLOCK, UNGROUPED_SIDES)
        for (sides in intArrayOf(1, group + 1)) {
            val left = DenseMatrix.wrap(order, sides, DoubleArray(order * sides))
            val right = DenseMatrix.wrap(sides, order, DoubleArray(sides * order))
            for (unitDiag in booleanArrayOf(false, true)) {
                assertAllocationFree(
                    "trsm left sides=$sides unit=$unitDiag",
                    TRIANGULAR_WARMUP,
                    TRIANGULAR_ITERATIONS,
                ) {
                    left.values.fill(1.0)
                    engine.trsm(triangle, left, lower = true, unitDiag = unitDiag, workspace = workspace)
                    left.values[0]
                }
                assertAllocationFree(
                    "trmm left sides=$sides unit=$unitDiag",
                    TRIANGULAR_WARMUP,
                    TRIANGULAR_ITERATIONS,
                ) {
                    left.values.fill(1.0)
                    engine.trmm(triangle, left, lower = true, unitDiag = unitDiag, workspace = workspace)
                    left.values[0]
                }
                assertAllocationFree(
                    "trsm right sides=$sides unit=$unitDiag",
                    TRIANGULAR_WARMUP,
                    TRIANGULAR_ITERATIONS,
                ) {
                    right.values.fill(1.0)
                    engine.trsm(
                        triangle,
                        right,
                        lower = true,
                        unitDiag = unitDiag,
                        right = true,
                        workspace = workspace,
                    )
                    right.values[0]
                }
                assertAllocationFree(
                    "trmm right sides=$sides unit=$unitDiag",
                    TRIANGULAR_WARMUP,
                    TRIANGULAR_ITERATIONS,
                ) {
                    right.values.fill(1.0)
                    engine.trmm(
                        triangle,
                        right,
                        lower = true,
                        unitDiag = unitDiag,
                        right = true,
                        workspace = workspace,
                    )
                    right.values[0]
                }
            }
        }
    }

    /**
     * The product tile and the products around it, at a depth that fits one block and one that does not.
     *
     * A tile holds its accumulators in vector registers for the whole depth of a block, so a compilation
     * that declined to scalar-replace one of them would allocate once per depth step, and a short depth is
     * where that is least likely to be hidden. The complete products are here for the same reason the Level
     * 2 callers are: the scheduling around a tile has packing, staging and a workspace loan in it, none of
     * which the raw tile can show.
     *
     * Fewer iterations than a panel takes, because one call of these is the arithmetic of many panels.
     */
    private fun checkProducts(engine: KoblasEngine) {
        val products = engine.productKernels
        require(products === SimdProductKernels) { "product allocation check selected ${products.name}" }
        val rows = products.tileRows
        val columns = products.tileColumns
        val shallow = 4
        val deep = 256
        val leftShallow = DoubleArray(rows * shallow) { 1.0 + (it % 13) * 0.125 }
        val rightShallow = DoubleArray(columns * shallow) { 0.5 + (it % 7) * 0.25 }
        val leftDeep = DoubleArray(rows * deep) { 1.0 + (it % 13) * 0.125 }
        val rightDeep = DoubleArray(columns * deep) { 0.5 + (it % 7) * 0.25 }
        val tile = DoubleArray(rows * columns)

        assertAllocationFree("product tile at depth $shallow") {
            products.productBlock(
                0.875, leftShallow, 0, rows * shallow, rightShallow, 0, columns * shallow,
                rows, columns, shallow, -0.25, tile, 0, rows,
            )
            tile[0]
        }
        assertAllocationFree("product tile at depth $deep") {
            products.productBlock(
                0.875, leftDeep, 0, rows * deep, rightDeep, 0, columns * deep,
                rows, columns, deep, -0.25, tile, 0, rows,
            )
            tile[0]
        }
        checkShortEdges(products, rows, columns)
        checkBlockOfTiles(products, rows, columns)
        checkWholeProducts(engine)
    }

    /**
     * The two edges a destination can have, each on its own.
     *
     * A destination whose rows stop part way through a lane block takes a different writeback from one whose
     * rows fill it, and one whose columns stop early stores fewer of them; neither is what a probe over
     * whole tiles exercises. Both are here because the writeback is where a compilation that stopped keeping
     * the accumulators in registers shows up, and the edges are the paths a full tile never takes.
     */
    private fun checkShortEdges(products: DenseProductKernels, tileRows: Int, tileColumns: Int) {
        val depth = 32
        val left = DoubleArray(tileRows * depth) { 1.0 + (it % 13) * 0.125 }
        val right = DoubleArray(tileColumns * depth) { 0.5 + (it % 7) * 0.25 }
        val target = DoubleArray(tileRows * tileColumns)
        val shortRows = tileRows - 1
        val shortColumns = tileColumns - 1

        assertAllocationFree("product tile with $shortRows of $tileRows rows") {
            products.productBlock(
                0.875, left, 0, tileRows * depth, right, 0, tileColumns * depth,
                shortRows, tileColumns, depth, -0.25, target, 0, tileRows,
            )
            target[0]
        }
        assertAllocationFree("product tile with $shortColumns of $tileColumns columns") {
            products.productBlock(
                0.875, left, 0, tileRows * depth, right, 0, tileColumns * depth,
                tileRows, shortColumns, depth, -0.25, target, 0, tileRows,
            )
            target[0]
        }
        assertAllocationFree("product tile short on both axes") {
            products.productBlock(
                0.875, left, 0, tileRows * depth, right, 0, tileColumns * depth,
                shortRows, shortColumns, depth, -0.25, target, 0, tileRows,
            )
            target[0]
        }
    }

    /**
     * A block of many tiles, which is where a tile is reached from inside a loop rather than on its own.
     *
     * A microkernel measured by itself can be allocation-free while the same code inlined into a traversal
     * is not, because how much a compiler will keep in registers depends on how large the method it is
     * compiling became. This is the smallest caller that has that shape.
     */
    private fun checkBlockOfTiles(products: DenseProductKernels, tileRows: Int, tileColumns: Int) {
        val depth = 64
        // One row and one column short of whole tiles, so the block has both edges inside it as well as the
        // full tiles that surround them.
        val rows = tileRows * 6 - 1
        val columns = tileColumns * 6 - 1
        val left = DoubleArray((rows + tileRows) * depth) { 1.0 + (it % 13) * 0.125 }
        val right = DoubleArray((columns + tileColumns) * depth) { 0.5 + (it % 7) * 0.25 }
        val target = DoubleArray(rows * columns)

        assertAllocationFree("block of tiles with short edges", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            products.productBlock(
                0.875, left, 0, tileRows * depth, right, 0, tileColumns * depth,
                rows, columns, depth, -0.25, target, 0, rows,
            )
            target[0]
        }
    }

    private fun checkWholeProducts(engine: KoblasEngine) {
        val workspace = Workspace()
        // Both shapes have to be ones this machine's tile geometry really packs, or the deep probe would
        // measure the unpacked route while claiming to measure depth blocks.
        val narrow = maxOf(engine.productKernels.tileRows + 1, PRODUCT_NARROW)
        require(engine.productKernels.packsProduct(narrow, narrow, PRODUCT_DEPTH)) {
            "the deep product probe at ${narrow}x${narrow}x$PRODUCT_DEPTH is not packed on this machine"
        }
        require(engine.productKernels.packsProduct(PRODUCT_ORDER, PRODUCT_ORDER, PRODUCT_ORDER)) {
            "the square product probe at $PRODUCT_ORDER cubed is not packed on this machine"
        }
        val square = PRODUCT_ORDER * PRODUCT_ORDER
        val wide = DenseMatrix.wrap(PRODUCT_ORDER, PRODUCT_ORDER, DoubleArray(square) { 1.0 })
        val wideTarget = DenseMatrix.wrap(PRODUCT_ORDER, PRODUCT_ORDER, DoubleArray(square))
        val deepLeft = DenseMatrix.wrap(
            narrow,
            PRODUCT_DEPTH,
            DoubleArray(narrow * PRODUCT_DEPTH) { 1.0 },
        )
        val deepRight = DenseMatrix.wrap(
            PRODUCT_DEPTH,
            narrow,
            DoubleArray(PRODUCT_DEPTH * narrow) { 0.5 },
        )
        val deepTarget = DenseMatrix.wrap(narrow, narrow, DoubleArray(narrow * narrow))
        val left = engine.packLeft(wide, transpose = false)
        val right = engine.packRight(wide, transpose = false)

        assertAllocationFree("gemm over one depth block", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, wide, false, wide, false, -0.25, wideTarget, workspace)
            wideTarget.values[0]
        }
        assertAllocationFree("gemm over several depth blocks", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, deepLeft, false, deepRight, false, -0.25, deepTarget, workspace)
            deepTarget.values[0]
        }
        assertAllocationFree("gemm over retained panels", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, left, right, -0.25, wideTarget)
            wideTarget.values[0]
        }
        checkUnpackedProducts(engine, workspace)
    }

    /**
     * The route a product too small to pay for packing takes, in both of its forms.
     *
     * The accumulating form borrows a destination column and the reducing form may borrow a coefficient
     * column to gather a strided one into; both come from the workspace, so a repeated call over one shape
     * asks for the same buffers. Neither reaches a product tile, and both are what a skinny operand gets,
     * so they belong beside the blocked probes rather than instead of them.
     */
    private fun checkUnpackedProducts(engine: KoblasEngine, workspace: Workspace) {
        val rows = 256
        val depth = 512
        val a = DenseMatrix.wrap(rows, depth, DoubleArray(rows * depth) { 1.0 + (it % 13) * 0.125 })
        val transposed = DenseMatrix.wrap(depth, rows, DoubleArray(depth * rows) { 1.0 + (it % 13) * 0.125 })
        val plainRight = DenseMatrix.wrap(depth, 2, DoubleArray(depth * 2) { 0.5 + (it % 7) * 0.25 })
        val storedRight = DenseMatrix.wrap(2, depth, DoubleArray(2 * depth) { 0.5 + (it % 7) * 0.25 })
        val target = DenseMatrix.wrap(rows, 2, DoubleArray(rows * 2))

        assertAllocationFree("unpacked product accumulating columns", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, a, false, plainRight, false, -0.25, target, workspace)
            target.values[0]
        }
        assertAllocationFree("unpacked product reducing columns", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, transposed, true, plainRight, false, -0.25, target, workspace)
            target.values[0]
        }
        assertAllocationFree("unpacked product over a gathered column", PRODUCT_WARMUP, PRODUCT_ITERATIONS) {
            engine.gemm(1e-12, transposed, true, storedRight, true, -0.25, target, workspace)
            target.values[0]
        }
    }

    /** The complete callers, where a staging copy or a wrapper would show up that a raw panel cannot. */
    private fun checkWholeOperations(engine: KoblasEngine) {
        val matrix = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until ORDER) triangle.values[i + i * ORDER] = 4.0
        val x = DoubleArray(ORDER) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(ORDER) { 0.5 }
        val target = DoubleArray(ORDER) { 1.0 }
        val vector = DenseVector.wrap(x)

        assertAllocationFree("gemv") {
            engine.gemv(0.875, matrix, x, -0.25, y)
            y[0]
        }
        assertAllocationFree("gemv transposed") {
            engine.gemv(0.875, matrix, x, -0.25, y, transpose = true)
            y[0]
        }
        assertAllocationFree("symv") {
            engine.symv(0.875, matrix, x, -0.25, y)
            y[0]
        }
        assertAllocationFree("ger") {
            engine.ger(1e-12, x, y, matrix)
            matrix.values[0]
        }
        assertAllocationFree("syr") {
            engine.syr(1e-12, vector, matrix)
            matrix.values[0]
        }
        assertAllocationFree("trmv") {
            target.fill(1.0)
            engine.trmv(triangle, target, lower = true)
            target[0]
        }
        assertAllocationFree("trsv") {
            target.fill(1.0)
            engine.trsv(triangle, target, lower = true)
            target[0]
        }
    }

    private fun assertAllocationFree(
        name: String,
        warmup: Int = WARMUP_ITERATIONS,
        iterations: Int = MEASUREMENT_ITERATIONS,
        block: Work,
    ) {
        val bytes = bytesPerIteration(block, warmup, iterations)
        check(bytes <= MAX_BYTES_PER_CALL) { "$name allocated $bytes B per call" }
        println("$name allocated $bytes B per call")
    }

    /**
     * One measured call.
     *
     * A named interface rather than a function type, because a `() -> Double` returns its result boxed and
     * that box is charged to whatever is being measured. This one compiles to a primitive return, so the
     * number is the call's own allocation and nothing else.
     */
    private fun interface Work {
        fun run(): Double
    }

    private fun bytesPerIteration(block: Work, warmup: Int, iterations: Int): Double {
        repeat(warmup) { resultSink = block.run() }
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(MEASUREMENT_WINDOWS) {
            val before = allocationBean.getThreadAllocatedBytes(id)
            repeat(iterations) { resultSink = block.run() }
            val after = allocationBean.getThreadAllocatedBytes(id)
            best = minOf(best, (after - before).toDouble() / iterations)
        }
        return best
    }
}
