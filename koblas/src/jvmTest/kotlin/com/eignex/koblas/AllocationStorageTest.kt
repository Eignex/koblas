package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Checks storage budgets, including gathers, result arrays and overlapping views.
 * Fresh sparse results allocate their structure; those probes bound additional scratch and copies.
 */
class AllocationStorageTest {
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

}
