package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Checks allocation-free Level 1 and sparse primitives, and matrix calls with warmed workspaces.
 * Fresh sparse results allocate their structure; those probes bound additional scratch and copies.
 */
class AllocationFreeTest {
    @Test
    fun `triplet construction reuses cursor and full result arrays`() {
        val rows = 128
        val columns = 4
        val unique = 32
        for (copies in intArrayOf(1, 2)) {
            val count = copies * unique
            val indices = IntArray(count) { (it / copies % 8) * 4 }
            val columnIndices = IntArray(count) { it / copies / 8 }
            val values = DoubleArray(count) { 1.0 + it * 0.125 }
            val rowScratch = ARRAY_HEADER_BYTES + (rows + 1) * Int.SIZE_BYTES + 4
            val pointers = ARRAY_HEADER_BYTES + (columns + 1) * Int.SIZE_BYTES + 4
            val runs = 2 * (2 * ARRAY_HEADER_BYTES + count * (Int.SIZE_BYTES + Double.SIZE_BYTES))
            val compact = if (copies == 1) 0 else 2 * ARRAY_HEADER_BYTES + unique * (Int.SIZE_BYTES + Double.SIZE_BYTES)
            // Two run descriptors, the result object, alignment and the probe's fixed allowance.
            val budget = rowScratch + pointers + runs + compact + 2 * FLOOR_BYTES

            val bytes = bytesPerIteration(1_000, budget) {
                SparseMatrix.ofTriplets(rows, columns, indices, columnIndices, values)
            }

            assertTrue(bytes <= budget, "triplet copies=$copies allocated $bytes B per call against $budget")
        }
    }

    @Test
    fun `overlapping contiguous copy allocates nothing`() {
        val n = 512
        val backing = DoubleArray(n + 1) { it * 0.01 }
        val source = StridedVector(backing, 0, n)
        val destination = StridedVector(backing, 1, n)

        val bytes = bytesPerIteration(1_000, FLOOR_BYTES) {
            copy(source, destination)
            backing
        }

        assertTrue(bytes <= FLOOR_BYTES, "overlapping copy allocated $bytes B per call")
    }

    @Test
    fun `disjoint shared buffer vector updates allocate nothing`() {
        val n = 512
        val backing = DoubleArray(2 * n) { it * 0.01 }
        val source = StridedVector(backing, 0, n, 2)
        val destination = StridedVector(backing, 1, n, 2)
        for (operation in listOf<() -> Unit>(
            { copy(source, destination) },
            { destination.axpy(1e-12, source) },
            { swap(source, destination) },
        )) {
            val bytes = bytesPerIteration(1_000, FLOOR_BYTES) {
                operation()
                backing
            }

            assertTrue(bytes <= FLOOR_BYTES, "shared buffer update allocated $bytes B per call")
        }
    }

    @Test
    fun `sparse dot against a dense view allocates nothing`() {
        val n = 512
        val backing = DoubleArray(2 * n) { it * 0.01 }
        val sparse = SparseVector.wrap(n, intArrayOf(0, 7, n - 1), doubleArrayOf(1.5, 0.0, -2.0))
        for (stride in intArrayOf(-2, -1, 1, 2)) {
            val view = StridedVector(backing, if (stride < 0) 2 * n - 2 else 1, n, stride)

            val bytes = bytesPerIteration(1_000, FLOOR_BYTES) { sparse dot view }

            assertTrue(bytes <= FLOOR_BYTES, "sparse view dot allocated $bytes B per call")
        }
    }

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        /** An array object's own bytes, beside the elements it holds. */
        const val ARRAY_HEADER_BYTES = 16

        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
    }

    @Test
    fun `dense level one kernels allocate nothing`() {
        val n = 512
        val a = DoubleArray(n) { it * 0.01 }
        val b = DoubleArray(n) { 1.0 / (it + 1) }
        val kernels = engine.vectorKernels

        val dot = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.dot(a, 0, b, 0, n) }
        val axpy = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.axpy(b, 0, 0.5, a, 0, n) }
        val nrm2 = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.nrm2(a, 0, n) }
        val strided = bytesPerIteration(1_000, FLOOR_BYTES) { kernels.dot(a, 0, b, 0, n / 2, 2, 2) }

        assertTrue(dot <= FLOOR_BYTES, "dot allocated $dot B per call")
        assertTrue(axpy <= FLOOR_BYTES, "axpy allocated $axpy B per call")
        assertTrue(nrm2 <= FLOOR_BYTES, "nrm2 allocated $nrm2 B per call")
        assertTrue(strided <= FLOOR_BYTES, "a strided dot allocated $strided B per call")
    }

    /**
     * The dense Level 2 callers a user writes, on the portable panels. Kover's instrumentation keeps HotSpot
     * from scalar-replacing a Vector API carrier, so the vector panels are measured by the
     * `simdDenseAllocationCheck` task instead; what this covers is the scheduling around them.
     */
    @Test
    fun `dense level two calls allocate nothing`() {
        val n = 48
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until n) triangle.values[i + i * n] = 4.0
        val x = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(n) { 0.5 }
        val view = DenseVector.wrap(x)
        val portable = BuiltinEngines.scalar

        val gemv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.gemv(1e-12, a, x, 1.0, y)
            y
        }
        val transposed = bytesPerIteration(400, FLOOR_BYTES) {
            portable.gemv(1e-12, a, x, 1.0, y, transpose = true)
            y
        }
        val symv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.symv(1e-12, a, x, 1.0, y)
            y
        }
        val ger = bytesPerIteration(400, FLOOR_BYTES) {
            portable.ger(1e-12, x, y, a)
            a
        }
        val syr = bytesPerIteration(400, FLOOR_BYTES) {
            portable.syr(1e-12, view, a)
            a
        }
        val trmv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.trmv(triangle, y, lower = true)
            y
        }
        val trsv = bytesPerIteration(400, FLOOR_BYTES) {
            portable.trsv(triangle, y, lower = true)
            y
        }

        assertTrue(gemv <= FLOOR_BYTES, "gemv allocated $gemv B per call")
        assertTrue(transposed <= FLOOR_BYTES, "transposed gemv allocated $transposed B per call")
        assertTrue(symv <= FLOOR_BYTES, "symv allocated $symv B per call")
        assertTrue(ger <= FLOOR_BYTES, "ger allocated $ger B per call")
        assertTrue(syr <= FLOOR_BYTES, "syr allocated $syr B per call")
        assertTrue(trmv <= FLOOR_BYTES, "trmv allocated $trmv B per call")
        assertTrue(trsv <= FLOOR_BYTES, "trsv allocated $trsv B per call")
    }

    /** The convenience callers, which pass a contiguous vector through as the array it already is. */
    @Test
    fun `contiguous convenience calls allocate nothing`() {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val contiguous = DenseVector.wrap(DoubleArray(n) { 1.0 + (it % 7) * 0.25 })
        val destination = DoubleArray(n)

        val gemv = bytesPerIteration(1_000, FLOOR_BYTES) {
            a.gemvInto(1e-12, contiguous, 1.0, destination)
            destination
        }
        val symv = bytesPerIteration(1_000, FLOOR_BYTES) {
            a.symvInto(1e-12, contiguous, 1.0, destination)
            destination
        }

        assertTrue(gemv <= FLOOR_BYTES, "gemvInto allocated $gemv B per call")
        assertTrue(symv <= FLOOR_BYTES, "symvInto allocated $symv B per call")
    }

    /**
     * A strided operand may require a gather array. The allocation bound also admits implementations
     * that address the stride directly.
     */
    @Test
    fun `a strided convenience operand is gathered once per call`() {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val buffer = DoubleArray(2 * n) { 1.0 + (it % 7) * 0.25 }
        val strided = StridedVector(buffer, 0, n, 2)
        val destination = DoubleArray(n)
        val oneGather = n * Double.SIZE_BYTES + ARRAY_HEADER_BYTES

        val gathered = bytesPerIteration(1_000, oneGather.toDouble()) {
            a.gemvInto(1e-12, strided, 1.0, destination)
            destination
        }

        assertTrue(gathered <= oneGather + FLOOR_BYTES, "a strided gemvInto allocated $gathered B per call")
    }

    /** A workspace lends the gather buffer so repeated strided calls allocate no fresh array. */
    @Test
    fun `a strided convenience operand lent a workspace allocates nothing`() {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val buffer = DoubleArray(2 * n) { 1.0 + (it % 7) * 0.25 }
        val strided = StridedVector(buffer, 0, n, 2)
        val destination = DoubleArray(n)
        val workspace = Workspace()
        a.gemvInto(1e-12, strided, 1.0, destination, workspace = workspace)

        val gathered = bytesPerIteration(1_000, FLOOR_BYTES) {
            a.gemvInto(1e-12, strided, 1.0, destination, workspace = workspace)
            destination
        }

        assertTrue(gathered <= FLOOR_BYTES, "a strided gemvInto with a workspace allocated $gathered B per call")
    }

    /**
     * The dense and sparse Level 2 calls whose operand is the buffer they write, which is the one case those
     * routines take scratch for. Without a workspace each snapshots into a fresh array per call.
     */
    @Test
    fun `aliased level two calls reuse a workspace`() {
        val n = 128
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val y = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf((j + 1) % n to 1.0 + j % 5) })
        val coefficients = sparse.values
        val workspace = Workspace()
        portable.gemv(1e-12, a, y, 1.0, y, workspace = workspace)
        koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, workspace = workspace)

        val dense = bytesPerIteration(400, FLOOR_BYTES) {
            portable.gemv(1e-12, a, y, 1.0, y, workspace = workspace)
            y
        }
        val symmetric = bytesPerIteration(400, FLOOR_BYTES) {
            portable.symv(1e-12, a, y, 1.0, y, workspace = workspace)
            y
        }
        val csc = bytesPerIteration(400, FLOOR_BYTES) {
            koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, workspace = workspace)
            coefficients
        }

        assertTrue(dense <= FLOOR_BYTES, "an aliased gemv with a workspace allocated $dense B per call")
        assertTrue(symmetric <= FLOOR_BYTES, "an aliased symv with a workspace allocated $symmetric B per call")
        assertTrue(csc <= FLOOR_BYTES, "an aliased sparse gemv with a workspace allocated $csc B per call")
    }

    /**
     * A dense Level 3 call lent a workspace. Without one these allocate by design: a solve gathers each
     * right-hand side and a multiply copies it again as its source.
     */
    @Test
    fun `dense level three calls reuse a workspace`() {
        val order = 32
        val sides = 8
        val portable = BuiltinEngines.scalar
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 0.25 + (it % 11) * 0.0625 })
        for (i in 0 until order) triangle.values[i + i * order] = 4.0
        val b = DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { 1.0 })
        val workspace = Workspace()
        portable.trsm(triangle, b, lower = true, workspace = workspace)
        portable.trmm(triangle, b, lower = true, workspace = workspace)

        val solve = bytesPerIteration(100, FLOOR_BYTES) {
            portable.trsm(triangle, b, lower = true, workspace = workspace)
            b
        }
        val multiply = bytesPerIteration(100, FLOOR_BYTES) {
            portable.trmm(triangle, b, lower = true, workspace = workspace)
            b
        }

        assertTrue(solve <= FLOOR_BYTES, "trsm with a workspace allocated $solve B per call")
        assertTrue(multiply <= FLOOR_BYTES, "trmm with a workspace allocated $multiply B per call")
    }

    /**
     * The matrix product on both of its routes, at a depth that fits one block and one that does not. The
     * blocked route's two packed panels are loans; a retained pair borrows nothing, having no copy left.
     */
    @Test
    fun `matrix products reuse a workspace at both depths`() {
        val portable = BuiltinEngines.scalar
        val workspace = Workspace()
        val wide = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48) { 1.0 + (it % 13) * 0.125 })
        val deepLeft = DenseMatrix.wrap(12, 400, DoubleArray(12 * 400) { 1.0 + (it % 7) * 0.25 })
        val deepRight = DenseMatrix.wrap(400, 12, DoubleArray(400 * 12) { 0.5 + (it % 5) * 0.125 })
        val deepTarget = DenseMatrix.wrap(12, 12, DoubleArray(144))
        val small = DenseMatrix.wrap(5, 5, DoubleArray(25) { 1.0 + it })
        val smallTarget = DenseMatrix.wrap(5, 5, DoubleArray(25))
        val left = portable.packLeft(wide, transpose = false)
        val right = portable.packRight(wide, transpose = false)
        val packedTarget = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48))
        portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
        portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)

        val shallow = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, workspace)
            packedTarget
        }
        val deep = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, workspace)
            deepTarget
        }
        val retained = bytesPerIteration(50, FLOOR_BYTES) {
            portable.gemm(1e-12, left, right, 1.0, packedTarget)
            packedTarget
        }
        portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
        val panelRoute = bytesPerIteration(200, FLOOR_BYTES) {
            portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, workspace)
            smallTarget
        }

        assertTrue(shallow <= FLOOR_BYTES, "a blocked product allocated $shallow B per call")
        assertTrue(deep <= FLOOR_BYTES, "a product over several depth blocks allocated $deep B per call")
        assertTrue(retained <= FLOOR_BYTES, "a retained product allocated $retained B per call")
        assertTrue(panelRoute <= FLOOR_BYTES, "an unpacked product allocated $panelRoute B per call")
    }

    /**
     * The one buffer an unpacked product takes when it is lent no workspace: the destination column it
     * accumulates before spending the multipliers, which is what keeps alpha on a sum of products.
     */
    @Test
    fun `an unpacked product without a workspace takes one destination column`() {
        val order = 64
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(order, 2, DoubleArray(order * 2) { 1.0 + (it % 13) * 0.125 })
        val b = DenseMatrix.wrap(2, 2, DoubleArray(4) { 0.5 })
        val c = DenseMatrix.wrap(order, 2, DoubleArray(order * 2))
        val oneColumn = order * Double.SIZE_BYTES + ARRAY_HEADER_BYTES

        val bytes = bytesPerIteration(500, oneColumn.toDouble()) {
            portable.gemm(1e-12, a, false, b, false, 1.0, c)
            c
        }

        assertTrue(bytes <= oneColumn + FLOOR_BYTES, "an unpacked product allocated $bytes B per call")
    }

    @Test
    fun `sparse vector kernels allocate nothing`() {
        val n = 128
        val vector = SparseVector.of(n, IntArray(n / 2) { it * 2 }, DoubleArray(n / 2) { it + 1.0 })
        val x = DoubleArray(n) { it * 0.01 }

        val levelOneBytes = bytesPerIteration(1_000, FLOOR_BYTES) { engine.sparseKernels.dot(vector, x) }

        assertTrue(levelOneBytes <= FLOOR_BYTES, "sparse level one allocated $levelOneBytes B per call")
    }

    @Test
    fun `a sparse transpose allocates only result storage`() {
        val rows = 128
        val columns = 4
        val a = SparseMatrix.ofColumns(
            rows,
            columns,
            List(columns) { j ->
                List(32) { i -> (i * 4 + j) to (i + 1.0) }
            },
        )
        val resultBytes = 3 * ARRAY_HEADER_BYTES + (rows + 1) * Int.SIZE_BYTES +
            a.nnz * (Int.SIZE_BYTES + Double.SIZE_BYTES)

        val bytes = bytesPerIteration(100, resultBytes + FLOOR_BYTES) { engine.transpose(a) }

        assertTrue(bytes <= resultBytes + FLOOR_BYTES, "sparse transpose allocated $bytes B per call")
    }

    @Test
    fun `a sparse addition with matching or disjoint patterns allocates only result storage`() {
        val rows = 128
        val columns = 4
        val a = SparseMatrix.ofColumns(
            rows,
            columns,
            List(columns) {
                List(32) { i -> (i * 4) to (i + 1.0) }
            },
        )
        for (overlap in booleanArrayOf(false, true)) {
            val b = SparseMatrix.ofColumns(
                rows,
                columns,
                List(columns) {
                    List(32) { i -> (i * 4 + if (overlap) 0 else 1) to (i + 0.5) }
                },
            )
            val entries = if (overlap) a.nnz else a.nnz + b.nnz
            val resultBytes = 3 * ARRAY_HEADER_BYTES + (columns + 1) * Int.SIZE_BYTES +
                entries * (Int.SIZE_BYTES + Double.SIZE_BYTES)

            val bytes = bytesPerIteration(200, resultBytes + FLOOR_BYTES) { engine.addScaled(0.5, a, false, b) }

            assertTrue(
                bytes <= resultBytes + FLOOR_BYTES,
                "sparse addition overlap=$overlap allocated $bytes B per call",
            )
        }
    }

    @Test
    fun `a zero alpha transposed sparse addition allocates only orientation and result storage`() {
        val rows = 128
        val columns = 4
        val a = SparseMatrix.ofColumns(
            rows,
            columns,
            List(columns) {
                List(32) { i -> (i * 4) to Double.NaN }
            },
        )
        val transposed = engine.transpose(a)
        val resultBytes = 3 * ARRAY_HEADER_BYTES + (columns + 1) * Int.SIZE_BYTES +
            a.nnz * (Int.SIZE_BYTES + Double.SIZE_BYTES)
        val budget = 2 * (resultBytes + FLOOR_BYTES)

        val bytes = bytesPerIteration(100, budget) { engine.addScaled(0.0, transposed, true, a) }

        assertTrue(bytes <= budget, "zero alpha transposed sparse addition allocated $bytes B per call")
    }

    @Test
    fun `sparse matrix-vector calls allocate nothing`() {
        val n = 128
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val x = DoubleArray(n) { it * 0.01 }
        val y = DoubleArray(n)

        val gemv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.gemv(1e-12, sparse, x, 1.0, y, transpose = true)
            y
        }
        val symv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.symv(1e-12, sparse, x, 1.0, y, lower = true)
            y
        }
        val trsv = bytesPerIteration(1_000, FLOOR_BYTES) {
            engine.trsv(sparse, y, lower = true)
            y
        }

        assertTrue(gemv <= FLOOR_BYTES, "sparse gemv allocated $gemv B per call")
        assertTrue(symv <= FLOOR_BYTES, "sparse symv allocated $symv B per call")
        assertTrue(trsv <= FLOOR_BYTES, "sparse trsv allocated $trsv B per call")
    }

    @Test
    fun `sparse dense destinations reuse a workspace`() {
        val n = 64
        val rightHandSides = 8
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val identity = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val solveBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val multiplyBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val transposed = DenseMatrix.zero(rightHandSides, n)
        val square = DenseMatrix.zero(n, n)
        val panel = DenseMatrix.zero(n, rightHandSides)
        val workspace = Workspace()
        // One warm call per shape, so the loans exist before the measured loop asks for them again.
        engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
        engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
        engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
        engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
        engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
        engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
        engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)

        val left = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, workspace = workspace)
            panel
        }
        val right = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, workspace = workspace)
            transposed
        }
        val rank = bytesPerIteration(300, FLOOR_BYTES) {
            engine.syrk(1e-12, sparse, false, 1.0, square, workspace = workspace)
            square
        }
        val symmetric = bytesPerIteration(300, FLOOR_BYTES) {
            engine.symm(1e-12, sparse, b, 1.0, panel, workspace = workspace)
            panel
        }
        val sparseDense = bytesPerIteration(300, FLOOR_BYTES) {
            engine.gemm(1e-12, sparse, false, identity, false, 1.0, square, workspace)
            square
        }
        val solve = bytesPerIteration(300, FLOOR_BYTES) {
            engine.trsm(sparse, solveBlock, lower = true, workspace = workspace)
            solveBlock
        }
        val multiply = bytesPerIteration(300, FLOOR_BYTES) {
            engine.trmm(sparse, multiplyBlock, lower = true, workspace = workspace)
            multiplyBlock
        }

        assertTrue(left <= FLOOR_BYTES, "left sparse gemm allocated $left B per call")
        assertTrue(right <= FLOOR_BYTES, "right transposed sparse gemm allocated $right B per call")
        assertTrue(rank <= FLOOR_BYTES, "sparse syrk allocated $rank B per call")
        assertTrue(symmetric <= FLOOR_BYTES, "sparse symm allocated $symmetric B per call")
        assertTrue(sparseDense <= FLOOR_BYTES, "direct sparse dense gemm allocated $sparseDense B per call")
        assertTrue(solve <= FLOOR_BYTES, "left sparse trsm allocated $solve B per call")
        assertTrue(multiply <= FLOOR_BYTES, "left sparse trmm allocated $multiply B per call")
    }
}
