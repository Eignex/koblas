package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engine.*
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.portable.sparse.AdjacentPreferringPanels
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotSame
import kotlin.test.assertSame
import kotlin.test.assertTrue

// The indexed dot is the only indexed operation with a vector body, and SparseSimdTest checks that body against
// the portable one at every short count and offset. The tests below therefore run the selected engine once,
// except where a shape that test leaves out, repeated support or an aliased operand, reaches the vector body.
@OptIn(UnsafeKoblasApi::class)
class SparseOperationsTest {
    private val vectorEngines: List<KoblasEngine> = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct()
    private val regionEngines: List<KoblasEngine> =
        listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()

    // Indexed vector operations.

    @Test
    fun `dot against a dense operand agrees with the scalar oracle around the vector minimum`() {
        for (nnz in listOf(1, 15, 16, 17, 33, 129)) {
            val rng = Random(nnz * 7919)
            val size = nnz * 8
            val x = randomSparseVector(size, nnz, rng)
            val y = randomVector(size, rng)
            val expected = BuiltinEngines.scalar.dot(x, y)

            val actual = koblas.dot(x, y)

            // A vectorized reduction sums lanes in a different order, so the bound scales with the count.
            assertTrue(
                abs(actual - expected) <= 1e-13 * nnz * (1.0 + abs(expected)),
                "nnz=$nnz: $actual vs $expected",
            )
        }
    }

    @Test
    fun `dot rejects a dense operand of the wrong length`() {
        val x = randomSparseVector(32, 4, Random(4))

        assertFailsWith<DimensionMismatch> { koblas.dot(x, DoubleArray(x.size + 1)) }
    }

    @Test
    fun `gather rejects a source that is its own destination`() {
        val x = SparseVector.of(4, intArrayOf(0, 1, 2, 3), doubleArrayOf(0.0, 1.0, 2.0, 3.0))

        assertFailsWith<IllegalArgumentException> { koblas.gather(x, x.values) }
        assertFailsWith<IllegalArgumentException> { koblas.gatherZero(x, x.values) }
        assertContentEquals(doubleArrayOf(0.0, 1.0, 2.0, 3.0), x.values)
    }

    @Test
    fun `stored value reductions call the selected dense kernels`() {
        for (count in listOf(0, 1, 3, 256)) {
            val x = SparseVector.of(1024, IntArray(count) { it * 4 }, DoubleArray(count) { 1.0 + it * 0.125 })
            val calls = mutableListOf<EngineImplementation>()
            fun record(operation: EngineImplementation, values: DoubleArray, offset: Int, length: Int, stride: Int) {
                assertSame(x.values, values)
                assertEquals(0, offset)
                assertEquals(count, length)
                assertEquals(1, stride)
                calls += operation
            }
            val selected = object : EngineKernels {
                override val name = "stored-reductions"
                override val implementations = setOf(EngineImplementation.Nrm2, EngineImplementation.Asum)
                override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(EngineImplementation.Nrm2, v, vOff, len, vStride)
                    return 0.0
                }

                override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                    record(EngineImplementation.Asum, v, vOff, len, vStride)
                    return 0.0
                }
            }
            val kernels = engineWithKernels(vectors = selected)

            kernels.nrm2(x)
            kernels.asum(x)

            assertEquals(listOf(EngineImplementation.Nrm2, EngineImplementation.Asum), calls, "count=$count")
        }
    }

    @Test
    fun `raw indexed operations honor independent offsets`() {
        val indices = intArrayOf(99, 1, 4, 6, 99)
        val values = doubleArrayOf(99.0, 99.0, 2.0, -3.0, 4.0, 99.0)
        val dense = doubleArrayOf(10.0, 5.0, 20.0, 30.0, 7.0, 50.0, -2.0)
        val updated = dense.copyOf()
        val scattered = DoubleArray(dense.size) { 8.0 }

        val dot = koblas.dot(indices, 1, values, 2, 3, dense)
        koblas.axpy(indices, 1, values, 2, 3, 0.5, updated)
        koblas.scatter(indices, 1, values, 2, 3, scattered)

        assertEquals(-19.0, dot)
        assertContentEquals(doubleArrayOf(10.0, 6.0, 20.0, 30.0, 5.5, 50.0, 0.0), updated)
        assertContentEquals(doubleArrayOf(8.0, 2.0, 8.0, 8.0, -3.0, 8.0, 4.0), scattered)
    }

    @Test
    fun `raw reductions permit repeated unsorted support`() {
        val indices = intArrayOf(2, 0, 2)
        val values = doubleArrayOf(3.0, 4.0, 5.0)
        val dense = doubleArrayOf(2.0, 7.0, -1.0)

        for (engine in vectorEngines) {
            assertEquals(0.0, engine.dot(indices, 0, values, 0, 3, dense), engine.name)
            assertEquals(sqrt(6.0), engine.nrm2(indices, 0, 3, dense), engine.name)
        }
    }

    @Test
    fun `raw empty slices at array ends are legal`() {
        val destination = doubleArrayOf(3.0)

        val dot = koblas.dot(intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0, DoubleArray(0))
        koblas.axpy(intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0, 2.0, destination)
        koblas.scatter(intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0, destination)
        val norm = koblas.nrm2(intArrayOf(9), 1, 0, DoubleArray(0))

        assertEquals(0.0, dot)
        assertEquals(0.0, norm)
        assertContentEquals(doubleArrayOf(3.0), destination)
    }

    @Test
    fun `axpy of a sparse vector with zero alpha does not evaluate values`() {
        val destination = doubleArrayOf(Double.NaN, -0.0)
        val poisoned = SparseVector.of(2, intArrayOf(0, 1), doubleArrayOf(Double.POSITIVE_INFINITY, Double.NaN))

        koblas.axpy(destination, -0.0, poisoned)

        assertTrue(destination[0].isNaN())
        assertEquals((-0.0).toBits(), destination[1].toBits())
    }

    @Test
    fun `raw scatter preserves explicit zeros and exceptional values`() {
        val source = doubleArrayOf(-0.0, Double.NaN, Double.NEGATIVE_INFINITY)
        val destination = DoubleArray(3)

        koblas.scatter(intArrayOf(0, 1, 2), 0, source, 0, 3, destination)

        assertEquals((-0.0).toBits(), destination[0].toBits())
        assertTrue(destination[1].isNaN())
        assertEquals(Double.NEGATIVE_INFINITY, destination[2])
    }

    @Test
    fun `indexed norm rescales finite values`() {
        val support = intArrayOf(1, 3)

        val large = koblas.nrm2(support, 0, 2, doubleArrayOf(0.0, 3e200, 0.0, 4e200))
        val small = koblas.nrm2(support, 0, 2, doubleArrayOf(0.0, 3e-200, 0.0, 4e-200))

        assertEquals(5e200, large, 1e188)
        assertEquals(5e-200, small, 1e-212)
    }

    @Test
    fun `container operations agree with raw indexed operations`() {
        val x = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val dense = doubleArrayOf(1.0, 5.0, 2.0, 3.0, 7.0, 4.0)
        val containerUpdate = dense.copyOf()
        val rawUpdate = dense.copyOf()
        val containerScatter = DoubleArray(6)
        val rawScatter = DoubleArray(6)

        koblas.axpy(containerUpdate, 0.25, x)
        koblas.axpy(x.indices, 0, x.values, 0, x.values.size, 0.25, rawUpdate)
        koblas.scatter(x, containerScatter)
        koblas.scatter(x.indices, 0, x.values, 0, x.values.size, rawScatter)

        assertEquals(koblas.dot(x, dense), koblas.dot(x.indices, 0, x.values, 0, x.values.size, dense))
        assertContentEquals(containerUpdate, rawUpdate)
        assertContentEquals(containerScatter, rawScatter)
        assertEquals(koblas.nrm2(x), koblas.nrm2(x.indices, 0, x.indices.size, containerScatter), 1e-15)
    }

    // Column dots.

    @Test
    fun `column dots agree with the scalar oracle across support lengths and offsets`() {
        for (count in listOf(0, 1, 17)) {
            val rows = count * 2 + 3
            val indices = intArrayOf(0, 1, 2) + IntArray(count) { 2 * it + 1 } + intArrayOf(rows - 1)
            val values = DoubleArray(indices.size) { (it % 17 - 8) * 0.125 }
            val matrix = SparseMatrix.wrap(
                rows,
                5,
                intArrayOf(0, 0, 3, 3 + count, 4 + count, 4 + count),
                indices,
                values,
            )
            val dense = DoubleArray(rows) { (it % 23 - 11) * 0.0625 }
            for (column in 0 until matrix.cols) {
                val start = matrix.colPointers[column]
                val length = matrix.colPointers[column + 1] - start
                val expected = BuiltinEngines.scalar.dot(indices, start, values, start, length, dense)

                for (selected in vectorEngines) {
                    val actual = selected.dotColumn(matrix, column, dense)

                    assertDotAgreesWithReference(expected, actual, "${selected.name} count=$count column=$column")
                }
            }
        }
    }

    @Test
    fun `column dot lends the original arrays at the selected offsets to the indexed leaf`() {
        val matrix = SparseMatrix.wrap(
            5,
            4,
            intArrayOf(0, 1, 3, 3, 3),
            intArrayOf(4, 0, 3),
            doubleArrayOf(9.0, 2.0, -3.0),
        )
        val denseOperand = DoubleArray(matrix.rows)
        val calls = mutableListOf<Pair<Int, Int>>()
        val selected = object : EngineKernels {
            override val implementations = setOf(EngineImplementation.IndexedDot)
            override val name: String = "test-SparseOperationsTest"

            override fun dot(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                assertSame(matrix.rowIndices, indices)
                assertSame(matrix.values, values)
                assertSame(dense, denseOperand)
                assertEquals(indexOffset, valueOffset)
                calls += indexOffset to count
                return 0.0
            }
        }
        val kernels = engineWithKernels(indexed = selected)

        for (column in 0 until matrix.cols) kernels.dotColumn(matrix, column, denseOperand)

        assertEquals(listOf(0 to 1, 1 to 2, 3 to 0, 3 to 0), calls)
    }

    @Test
    fun `column dots evaluate stored zeros and skip absent nonfinite positions`() {
        for (stored in listOf(0.0, -0.0, 2.0, Double.POSITIVE_INFINITY, Double.NaN)) {
            val matrix = SparseMatrix.wrap(
                4,
                3,
                intArrayOf(0, 1, 2, 3),
                intArrayOf(0, 2, 3),
                doubleArrayOf(Double.NaN, stored, Double.NaN),
            )
            for (rhs in listOf(3.0, Double.POSITIVE_INFINITY, Double.NaN)) {
                val dense = doubleArrayOf(Double.NaN, Double.NEGATIVE_INFINITY, rhs, Double.NaN)
                val expected = BuiltinEngines.scalar.dot(matrix.rowIndices, 1, matrix.values, 1, 1, dense)

                val actual = koblas.dotColumn(matrix, 1, dense)

                assertDotAgreesWithReference(expected, actual, "stored=$stored rhs=$rhs")
            }
        }
    }

    @Test
    fun `column dots see coefficient updates without retaining a snapshot`() {
        val matrix = SparseMatrix.wrap(
            4,
            3,
            intArrayOf(0, 1, 3, 4),
            intArrayOf(3, 0, 2, 1),
            doubleArrayOf(8.0, 2.0, -3.0, 9.0),
        )
        val dense = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        for (coefficient in listOf(2.0, 5.0, -0.0)) {
            matrix.values[1] = coefficient
            val expected = BuiltinEngines.scalar.dot(matrix.rowIndices, 1, matrix.values, 1, 2, dense)

            val actual = koblas.dotColumn(matrix, 1, dense)

            assertDotAgreesWithReference(expected, actual, "coefficient=$coefficient")
        }
    }

    @Test
    fun `column dots allow the coefficient buffer as the dense operand without mutation`() {
        val matrix = SparseMatrix.wrap(
            4,
            3,
            intArrayOf(0, 1, 3, 4),
            intArrayOf(3, 0, 2, 1),
            doubleArrayOf(8.0, 2.0, -3.0, 9.0),
        )
        val before = matrix.values.copyOf()
        val expected = BuiltinEngines.scalar.dot(matrix.rowIndices, 1, before, 1, 2, before)

        for (engine in vectorEngines) {
            val actual = engine.dotColumn(matrix, 1, matrix.values)

            assertDotAgreesWithReference(expected, actual, engine.name)
            assertContentEquals(before, matrix.values, engine.name)
        }
    }

    @Test
    fun `column dots accept empty columns with no rows`() {
        val matrix = SparseMatrix.wrap(0, 2, intArrayOf(0, 0, 0), IntArray(0), DoubleArray(0))
        val dense = DoubleArray(0)
        val expected = BuiltinEngines.scalar.dot(IntArray(0), 0, DoubleArray(0), 0, 0, dense)

        for (column in 0 until matrix.cols) {
            assertDotAgreesWithReference(expected, koblas.dotColumn(matrix, column, dense), "column=$column")
        }
    }

    @Test
    fun `column dots reject invalid columns including a matrix with no columns`() {
        for (columns in listOf(0, 2)) {
            val matrix = SparseMatrix.wrap(3, columns, IntArray(columns + 1), IntArray(0), DoubleArray(0))
            for (column in listOf(Int.MIN_VALUE, -1, columns, Int.MAX_VALUE)) {
                assertFailsWith<IndexOutOfBoundsException>("columns=$columns column=$column") {
                    koblas.dotColumn(matrix, column, DoubleArray(matrix.rows))
                }
            }
        }
    }

    @Test
    fun `column dots reject dense shape mismatch even for empty support`() {
        val matrix = SparseMatrix.wrap(3, 1, intArrayOf(0, 0), IntArray(0), DoubleArray(0))
        for (length in listOf(0, 2, 4)) {
            assertFailsWith<IllegalArgumentException>("length=$length") {
                koblas.dotColumn(matrix, 0, DoubleArray(length))
            }
        }
    }

    // Aliased operands: a destination may share an input's buffer, and the call must agree with one made on
    // independent buffers holding the values present at entry.

    @Test
    fun `gemv reads the input vector it was given when that vector is the destination`() {
        val values = doubleArrayOf(2.0, 3.0, 0.0, 4.0)
        for (transpose in booleanArrayOf(false, true)) {
            val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), values.copyOf())
            val expected = doubleArrayOf(1.0, 2.0)
            koblas.gemv(1.0, a, doubleArrayOf(1.0, 2.0), 0.0, expected, transpose)

            val shared = doubleArrayOf(1.0, 2.0)
            koblas.gemv(1.0, a, shared, 0.0, shared, transpose)

            assertContentEquals(expected, shared, "transpose=$transpose")
        }
    }

    @Test
    fun `gemv reads the coefficients it was given when they are the destination`() {
        for (transpose in booleanArrayOf(false, true)) {
            val x = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            val independent = permutation()
            val expected = independent.values.copyOf()
            koblas.gemv(0.5, independent, x, -1.5, expected, transpose)

            val shared = permutation()
            koblas.gemv(0.5, shared, x, -1.5, shared.values, transpose)

            assertContentEquals(expected, shared.values, "transpose=$transpose")
        }
    }

    @Test
    fun `symv reads the vector and coefficients it was given when both are the destination`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(2.0, 5.0))
        val shared = a.values

        koblas.symv(1.0, a, shared, 0.0, shared, lower = true)

        assertContentEquals(doubleArrayOf(4.0, 25.0), shared)
    }

    @Test
    fun `symm reads the operands it was given when all three share one buffer`() {
        val allShared = doubleArrayOf(2.0, 3.0, 7.0, 5.0)
        val sharedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), allShared)
        val sharedB = DenseMatrix.wrap(2, 2, allShared)
        val sharedC = DenseMatrix.wrap(2, 2, allShared)

        koblas.symm(1.0, sharedA, sharedB, 0.0, sharedC, lower = true, scratch = Scratch())

        assertContentEquals(doubleArrayOf(13.0, 21.0, 29.0, 46.0), sharedC.values)
    }

    @Test
    fun `a sparse pair product reads the coefficients it was given when they are the destination`() {
        val aliasedValues = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val aliasedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), aliasedValues)
        val identity = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val aliasedC = DenseMatrix.wrap(2, 2, aliasedValues)

        koblas.gemm(1.0, aliasedA, false, identity, false, 0.0, aliasedC, scratch = Scratch())

        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0, 7.0), aliasedC.values)
    }

    @Test
    fun `trsv solves against the triangle it was given when the right hand side is its values`() {
        // A column whose diagonal is missing, so the solve divides by zero and every later column depends on
        // a coefficient the substitution would already have overwritten.
        val independent = aliasTriangle()
        val expected = independent.values.copyOf()
        koblas.trsv(independent, expected, lower = true)

        val shared = aliasTriangle()
        koblas.trsv(shared, shared.values, lower = true)

        assertContentEquals(expected, shared.values)
    }

    @Test
    fun `every triangular side and direction reads the triangle it was given`() {
        val coefficients = doubleArrayOf(3.0, -0.5, 0.75, 2.5, -1.25, 4.0)
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unitDiag in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        for (solve in booleanArrayOf(false, true)) {
                            val what = "lower=$lower transpose=$transpose unit=$unitDiag right=$right solve=$solve"
                            val rows = if (right) 2 else 3
                            val columns = if (right) 3 else 2
                            val independent = filledTriangle(lower, coefficients)
                            val expected = DenseMatrix.wrap(rows, columns, coefficients.copyOf())
                            val shared = filledTriangle(lower, coefficients)
                            val block = DenseMatrix.wrap(rows, columns, shared.values)
                            val scratch = Scratch()

                            if (solve) {
                                koblas.trsm(independent, expected, lower, transpose, unitDiag, right, -0.75)
                                koblas.trsm(shared, block, lower, transpose, unitDiag, right, -0.75, scratch = scratch)
                            } else {
                                koblas.trmm(independent, expected, lower, transpose, unitDiag, right, -0.75)
                                koblas.trmm(shared, block, lower, transpose, unitDiag, right, -0.75, scratch = scratch)
                            }

                            assertClose(expected, block, what, tolerance = 1e-9)
                        }
                    }
                }
            }
        }
    }

    // Alias snapshots borrow from the workspace. The tests above check their numerical results, which do not
    // depend on whether a snapshot was allocated or borrowed, so these check only the loans.

    @Test
    fun `every level two routine stages its alias from the workspace`() {
        val n = 4

        assertStaged(n, "gemv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.gemv(0.5, permutation(), y, -1.5, y, false, scratch = w)
        }
        assertStaged(n, "gemv over the coefficients it writes") { w ->
            val a = permutation()
            koblas.gemv(0.5, a, doubleArrayOf(1.0, -2.0, 0.5, 3.0), -1.5, a.values, false, scratch = w)
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.symv(0.5, permutation(), y, -1.5, y, true, scratch = w)
        }
        assertStaged(3, "trsv over the coefficients that are its right-hand side") { w ->
            val a = aliasTriangle()
            koblas.trsv(a, a.values, lower = true, scratch = w)
        }
        assertStaged(3, "trmv over the coefficients that are its right-hand side") { w ->
            val a = aliasTriangle()
            koblas.trmv(a, a.values, lower = true, scratch = w)
        }
    }

    /** Sparse staging copies only coefficients: structural arrays cannot alias a dense destination. */
    @Test
    fun `a staged sparse operand copies its coefficients and shares its structure`() {
        val workspace = Scratch()
        val a = permutation()

        koblas.gemv(0.5, a, doubleArrayOf(1.0, -2.0, 0.5, 3.0), -1.5, a.values, false, scratch = workspace)

        assertEquals(1, workspace.available(a.nnz), "the coefficients were not staged from the workspace")
        assertEquals(0, workspace.availableI32(a.cols + 1), "the column pointers were copied")
        assertEquals(0, workspace.availableI32(a.nnz), "the row indices were copied")
    }

    // Zero multipliers and empty shapes.

    @Test
    fun `symv with a zero alpha reads neither operand`() {
        val poisoned = SparseMatrix.wrap(1, 1, intArrayOf(0, 1), intArrayOf(0), doubleArrayOf(Double.NaN))
        val y = doubleArrayOf(Double.NaN)

        koblas.symv(0.0, poisoned, doubleArrayOf(Double.NaN), 0.0, y)

        assertEquals(0.0, y[0])
    }

    @Test
    fun `a dense rank update over an empty source still applies beta to the selected triangle`() {
        val empty = SparseMatrix.ofColumns(2, 0, emptyList())
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        koblas.syrk(1.0, empty, transpose = false, 2.0, destination, lower = true)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 3.0, 8.0), destination.values)
    }

    @Test
    fun `empty products retain requested shapes and scalar behavior`() {
        val a = SparseMatrix.ofColumns(2, 0, emptyList())
        val b = SparseMatrix.ofColumns(0, 3, List(3) { emptyList() })
        val dense = DenseMatrix.wrap(2, 3, DoubleArray(6) { 4.0 })
        val emptySymmetric = SparseMatrix.ofColumns(0, 0, emptyList())

        val sparse = koblas.gemm(2.0, a, false, b, false)
        koblas.gemm(0.0, a, false, b, false, -0.5, dense)
        koblas.symv(1.0, emptySymmetric, DoubleArray(0), 0.0, DoubleArray(0))
        koblas.symm(1.0, emptySymmetric, DenseMatrix.zero(0, 3), 0.0, DenseMatrix.zero(0, 3))

        assertEquals(2, sparse.rows)
        assertEquals(3, sparse.cols)
        assertEquals(0, sparse.nnz)
        assertTrue(dense.values.all { it == -2.0 })
    }

    // One extent is zero and the other is a million, which separates returning early from stepping through a
    // million empty panels or columns. The staging backend is the one that would borrow if the call went on.
    @Test
    fun `an empty destination stages nothing and borrows nothing`() {
        val wide = 1 shl 20
        val staging = engineWithKernels(panels = AdjacentPreferringPanels)
        for ((order, sides) in listOf(0 to wide, wide to 0)) {
            val a = SparseMatrix.wrap(order, order, IntArray(order + 1), IntArray(0), DoubleArray(0))
            for (right in booleanArrayOf(false, true)) {
                val b = if (right) {
                    DenseMatrix.wrap(sides, order, DoubleArray(0))
                } else {
                    DenseMatrix.wrap(order, sides, DoubleArray(0))
                }
                for (operation in listOf("gemm", "symm", "trsm", "trmm")) {
                    val workspace = Scratch()

                    when (operation) {
                        "gemm" -> staging.gemm(0.875, a, false, b, false, -0.25, b, right = right, scratch = workspace)
                        "symm" -> staging.symm(0.875, a, b, -0.25, b, lower = true, right = right, scratch = workspace)
                        "trsm" -> staging.trsm(a, b, lower = true, right = right, alpha = 0.875, scratch = workspace)
                        else -> staging.trmm(a, b, lower = true, right = right, alpha = 0.875, scratch = workspace)
                    }

                    val context = "$operation order=$order sides=$sides right=$right"
                    assertEquals(0, workspace.idleLengths(), "$context took a loan")
                    assertEquals(0, workspace.idleI32Lengths(), "$context took an index loan")
                }
            }
        }
    }

    @Test
    fun `a triangular call with no right hand sides still validates dimensions`() {
        val order = 3
        val triangles = listOf(
            SparseMatrix.wrap(order, order, IntArray(order + 1), IntArray(0), DoubleArray(0)),
            SparseMatrix.wrap(order, order + 1, IntArray(order + 2), IntArray(0), DoubleArray(0)),
        )
        for (a in triangles) {
            for (right in booleanArrayOf(false, true)) {
                for (solve in booleanArrayOf(false, true)) {
                    val dimension = if (a.rows == a.cols) order + 1 else order
                    val b = if (right) DenseMatrix.zero(0, dimension) else DenseMatrix.zero(dimension, 0)
                    val workspace = Scratch()

                    assertFailsWith<DimensionMismatch>("shape=${a.rows} by ${a.cols} right=$right solve=$solve") {
                        if (solve) {
                            koblas.trsm(a, b, lower = true, right = right, scratch = workspace)
                        } else {
                            koblas.trmm(a, b, lower = true, right = right, scratch = workspace)
                        }
                    }

                    assertEquals(0, workspace.idleLengths())
                    assertEquals(0, workspace.idleI32Lengths())
                }
            }
        }
    }

    // Allocating products.

    @Test
    fun `allocating gemv multiplies a matrix and its transpose`() {
        val a = SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(
                listOf(0 to 1.0),
                listOf(1 to 3.0),
                listOf(0 to 2.0),
            ),
        )
        assertContentEquals(doubleArrayOf(7.0, 9.0), koblas.gemv(a, doubleArrayOf(1.0, 3.0, 3.0)))
        assertContentEquals(doubleArrayOf(1.0, 6.0, 2.0), koblas.gemv(a, doubleArrayOf(1.0, 2.0), transpose = true))
    }

    // Shape validation.

    // The extents hold no entries, so a call that sized its result before checking the vector would fail on the
    // allocation rather than on the shape.
    @Test
    fun `allocating gemv rejects incompatible vectors before oversized results`() {
        val a = SparseMatrix.wrap(Int.MAX_VALUE, 0, intArrayOf(0), intArrayOf(), doubleArrayOf())
        for (transpose in booleanArrayOf(false, true)) {
            val x = if (transpose) Int.MAX_VALUE else 0
            val y = if (transpose) 0 else Int.MAX_VALUE

            val failure = assertFailsWith<DimensionMismatch> { koblas.gemv(a, doubleArrayOf(1.0), transpose) }

            assertEquals(
                "gemv: ${Int.MAX_VALUE}x0 with transpose=$transpose needs x=$x and y=$y, got x=1 and y=$y",
                failure.message,
            )
        }
    }

    @Test
    fun `a dense operand that does not meet the sparse one from the right is rejected`() {
        val rng = Random(20260918)
        val sparse = randomSparse(5, 4, rng)

        assertFailsWith<DimensionMismatch> {
            koblas.gemm(1.0, sparse, false, randomMatrix(3, 3, rng), false, 0.0, DenseMatrix.zero(3, 4), right = true)
        }
    }

    @Test
    fun `a second operand that does not meet the first is rejected`() {
        val rng = Random(20260902)
        val sparse = randomSparse(5, 4, rng)

        assertFailsWith<DimensionMismatch> { koblas.gemm(sparse, randomMatrix(3, 3, rng)) }
        assertFailsWith<DimensionMismatch> { koblas.gemm(sparse, randomSparse(3, 4, rng)) }
    }

    @Test
    fun `a destination of the wrong shape is rejected`() {
        val rng = Random(20260901)
        val sparse = randomSparse(5, 4, rng)
        val b = randomMatrix(4, 3, rng)

        assertFailsWith<DimensionMismatch> {
            koblas.gemm(1.0, sparse, false, b, false, 0.0, DenseMatrix.zero(5, 2))
        }
    }

    @Test
    fun `a right hand side that does not meet the triangle is rejected on both sides`() {
        val rng = Random(20260824)
        val n = 4
        val triangle = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 2.0) })

        assertFailsWith<DimensionMismatch> { koblas.trsm(triangle, randomMatrix(n + 1, 2, rng), lower = true) }
        assertFailsWith<DimensionMismatch> {
            koblas.trsm(triangle, randomMatrix(2, n + 1, rng), lower = true, right = true)
        }
    }

    // Scaled addition.

    @Test
    fun `scaled addition retains the structural union`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val b = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 7.0), listOf(1 to -3.0)))

        val sum = koblas.addScaled(1.0, a, false, b)

        assertContentEquals(intArrayOf(0, 2, 3), sum.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 1, 1), sum.copyRowIndices())
        assertContentEquals(doubleArrayOf(7.0, 7.0, 0.0), sum.values)
    }

    @Test
    fun `scaled addition with a zero alpha keeps the union without reading a`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to Double.NaN), listOf(1 to Double.NaN)))
        val b = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 5.0, 1 to 7.0), listOf(1 to -3.0)))
        val aOnly = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to Double.NaN)))
        val empty = SparseMatrix.ofColumns(1, 1, listOf(emptyList()))

        val zero = koblas.addScaled(-0.0, a, false, b)
        val unread = koblas.addScaled(-0.0, aOnly, false, empty)

        assertContentEquals(intArrayOf(0, 2, 3), zero.copyColumnPointers())
        assertEquals(5.0, zero[0, 0])
        assertEquals(7.0, zero[1, 0])
        assertEquals(-3.0, zero[1, 1])
        assertEquals(0.0, unread.values.single(), absoluteTolerance = 0.0)
    }

    @Test
    fun `scaled addition owns its buffers for matching and disjoint patterns`() {
        val a = SparseMatrix.ofColumns(4, 2, listOf(listOf(0 to 2.0), listOf(2 to 3.0)))
        for (overlap in booleanArrayOf(false, true)) {
            val b = SparseMatrix.ofColumns(
                4,
                2,
                listOf(listOf((if (overlap) 0 else 1) to 4.0), listOf((if (overlap) 2 else 3) to 5.0)),
            )

            val result = koblas.addScaled(0.5, a, false, b)

            for (operand in listOf(a, b)) {
                assertNotSame(operand.colPointers, result.colPointers)
                assertNotSame(operand.rowIndices, result.rowIndices)
                assertNotSame(operand.values, result.values)
            }
        }
    }

    @Test
    fun `scaled addition agrees for matching disjoint and overlapping patterns`() {
        val a = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to 2.0, 3 to 0.0), emptyList(), listOf(1 to -1.0), listOf(3 to 4.0)),
        )
        val matching = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to -2.0, 3 to 1.0), emptyList(), listOf(1 to 0.0), listOf(3 to 2.0)),
        )
        val disjoint = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(1 to 3.0), emptyList(), listOf(2 to 0.0), listOf(4 to -2.0)),
        )
        val overlapping = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to 3.0, 4 to 1.0), emptyList(), listOf(2 to 0.0), listOf(3 to -2.0)),
        )
        for (b in listOf(matching, disjoint, overlapping)) {
            for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
                val left = SparseMatrix.wrap(
                    a.rows,
                    a.cols,
                    a.copyColumnPointers(),
                    a.copyRowIndices(),
                    DoubleArray(a.nnz) { if (alpha == 0.0) Double.NaN else a.values[it] },
                )
                for (transpose in booleanArrayOf(false, true)) {
                    val source = if (transpose) ReferenceSparseBlas.transpose(left) else left
                    val expected = ReferenceSparseBlas.addScaled(alpha, source, transpose, b)

                    val actual = koblas.addScaled(alpha, source, transpose, b)

                    val context = "alpha=$alpha transpose=$transpose"
                    assertEquals(expected.rows, actual.rows, context)
                    assertEquals(expected.cols, actual.cols, context)
                    assertContentEquals(expected.copyColumnPointers(), actual.copyColumnPointers(), context)
                    assertContentEquals(expected.copyRowIndices(), actual.copyRowIndices(), context)
                    assertClose(expected.values, actual.values, context)
                }
            }
        }
    }

    // Region destinations: a view inside a larger parent, so every write outside the view shows as a change.

    @Test
    fun `sparse pair products leave empty padded destinations untouched`() {
        for (engine in regionEngines) {
            for ((rows, columns) in listOf(0 to 3, 3 to 0)) {
                for (size in intArrayOf(0, 3)) {
                    for (alpha in doubleArrayOf(0.0, 1.0)) {
                        for (beta in doubleArrayOf(0.0, 0.5, 1.0)) {
                            val a = SparseMatrix.ofColumns(rows, 1, listOf(emptyList()))
                            val b = SparseMatrix.ofColumns(1, columns, List(columns) { emptyList() })
                            val backing = DoubleArray(size) { -77.0 }
                            val before = backing.copyOf()
                            val c = DenseMatrixView.wrap(rows, columns, backing, size, rows + 1)

                            engine.gemm(alpha, a, false, b, false, beta, c)

                            assertContentEquals(before, backing, "${engine.name} rows=$rows columns=$columns")
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `products into a region retain dense operand addressing`() {
        val sparse = regionSource()
        for (engine in regionEngines) {
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    val context = "${engine.name} transposeA=$transposeA transposeB=$transposeB"
                    for (right in booleanArrayOf(false, true)) {
                        val b = region(2)
                        assertRegionCall("$context right=$right", {
                            ReferenceSparseBlas.gemm(0.875, sparse, transposeA, copy(b), transposeB, -0.25, it, right)
                        }) {
                            engine.gemm(0.875, sparse, transposeA, b, transposeB, -0.25, it, right = right)
                        }
                    }
                    assertRegionCall("$context sparse pair", {
                        ReferenceSparseBlas.gemm(0.875, sparse, transposeA, sparse, transposeB, -0.25, it)
                    }) {
                        engine.gemm(0.875, sparse, transposeA, sparse, transposeB, -0.25, it)
                    }
                }
            }
        }
    }

    @Test
    fun `structured routines into a region preserve parent entries`() {
        val a = regionSource()
        val dense = copy(a)
        val b = region(2)
        for (engine in regionEngines) {
            for (lower in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    assertRegionCall("${engine.name} symm lower=$lower right=$right", {
                        ReferenceSparseBlas.symm(0.875, a, copy(b), -0.25, it, lower, right)
                    }) {
                        engine.symm(0.875, a, b, -0.25, it, lower, right)
                    }
                }
                for (transpose in booleanArrayOf(false, true)) {
                    assertRegionCall("${engine.name} syrk lower=$lower transpose=$transpose", {
                        ReferenceSparseBlas.syrk(0.875, a, transpose, -0.25, it, lower)
                    }) {
                        engine.syrk(0.875, a, transpose, -0.25, it, lower)
                    }
                    for (right in booleanArrayOf(false, true)) {
                        val flags = "lower=$lower transpose=$transpose right=$right"
                        assertRegionCall("${engine.name} trsm $flags", {
                            ReferenceBlas.trsm(dense, it, lower, transpose, true, right, 0.875)
                        }) {
                            engine.trsm(a, it, lower, transpose, true, right, 0.875)
                        }
                        assertRegionCall("${engine.name} trmm $flags", {
                            ReferenceBlas.trmm(dense, it, lower, transpose, true, right, 0.875)
                        }) {
                            engine.trmm(a, it, lower, transpose, true, right, 0.875)
                        }
                    }
                }
            }
        }
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double, context: String) {
        if (expected.isFinite() && expected != 0.0) {
            assertTrue(abs(actual - expected) <= 1e-12 * (1.0 + abs(expected)), context)
        } else {
            assertEquals(expected.toBits(), actual.toBits(), context)
        }
    }

    private fun randomSparseVector(size: Int, nnz: Int, rng: Random): SparseVector {
        val stride = size / nnz
        val indices = IntArray(nnz) { k -> k * stride + rng.nextInt(stride) }
        return SparseVector.of(size, indices, DoubleArray(nnz) { rng.nextDouble(-1.0, 1.0) })
    }

    private fun randomSparse(rows: Int, cols: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        rows,
        cols,
        List(cols) { (0 until rows).filter { rng.nextDouble() < 0.35 }.map { it to rng.nextDouble(-1.0, 1.0) } },
    )

    /**
     * A square matrix storing one entry per column, so its coefficient array is exactly as long as the
     * vectors and either of them can be the buffer a call writes. The entries are off the diagonal, so the
     * two orientations of the product read different positions of it.
     */
    private fun permutation(): SparseMatrix = SparseMatrix.ofColumns(
        4,
        4,
        listOf(listOf(1 to 2.0), listOf(3 to -3.0), listOf(0 to 5.0), listOf(2 to 7.0)),
    )

    /**
     * A lower triangle storing as many coefficients as it has rows, which is what lets its values be the
     * right-hand side a triangular routine writes.
     */
    private fun aliasTriangle(): SparseMatrix =
        SparseMatrix.wrap(3, 3, intArrayOf(0, 2, 3, 3), intArrayOf(0, 2, 1), doubleArrayOf(2.0, 3.0, 4.0))

    /**
     * A triangle of order three whose stored entries fill its selected half exactly, so its coefficient array
     * is also a valid three-by-two block and the two really do share one buffer. Six entries is the smallest
     * size where a right-hand side and a left-hand one can both be formed from the same array.
     */
    private fun filledTriangle(lower: Boolean, values: DoubleArray): SparseMatrix = SparseMatrix.ofColumns(
        3,
        3,
        List(3) { j ->
            (0 until 3).mapNotNull { i ->
                val inTriangle = if (lower) i >= j else i <= j
                if (inTriangle) i to values[triangleSlot(i, j, lower)] else null
            }
        },
    )

    private fun triangleSlot(i: Int, j: Int, lower: Boolean): Int {
        var slot = 0
        for (column in 0 until 3) {
            for (row in 0 until 3) {
                val inTriangle = if (lower) row >= column else row <= column
                if (!inTriangle) continue
                if (row == i && column == j) return slot
                slot++
            }
        }
        return 0
    }

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Scratch) -> Unit) {
        val workspace = Scratch()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }

    private fun region(seed: Int): DenseMatrixView {
        val parent = DenseMatrix.wrap(9, 8, DoubleArray(72) { -77.0 })
        val result = parent.view(1, 1, 7, 6).view(1, 1, 5, 5)
        for (j in 0 until 5) for (i in 0 until 5) result[i, j] = (i + j + seed) / 17.0
        return result
    }

    private fun copy(a: Matrix): DenseMatrix = DenseMatrix.ofRows(a.toArray())

    private fun regionSource(): SparseMatrix = SparseMatrix.ofColumns(
        5,
        5,
        List(5) { j ->
            List(5) { i -> i to if (i == j) 4.0 else (i + j + 1) / 31.0 }
        },
    )

    /**
     * Runs [call] on a fresh region and [reference] on a dense copy of it, then checks the region's whole
     * backing array: the view's entries against the reference and every other entry of the parent unchanged.
     */
    private fun assertRegionCall(context: String, reference: (DenseMatrix) -> Unit, call: (DenseMatrixView) -> Unit) {
        val c = region(3)
        val before = c.values.copyOf()
        val expected = copy(c)
        reference(expected)

        call(c)

        for (j in 0 until c.cols) {
            for (i in 0 until c.rows) {
                before[c.offset + i + j * c.leadingDimension] = expected[i, j]
            }
        }
        assertClose(before, c.values, context)
    }
}
