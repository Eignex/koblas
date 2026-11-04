package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.assertIamaxHonoursItsContract
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.sparse.*
import kotlin.random.Random
import kotlin.test.*

class VectorOpsTest {
    @Test
    fun `identical mapping axpy agrees with the scalar snapshot reference`() {
        val inputs = doubleArrayOf(
            1.5e200,
            -1.5e200,
            0.75,
            -0.0,
            0.0,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.NaN,
        )
        for (alpha in doubleArrayOf(0.0, -0.0, 0.75, -1.0)) {
            for ((source, destination) in identicalDenseMappings()) {
                for (i in 0 until source.size) source[i] = inputs[i % inputs.size]
                val expected = source.values.copyOf()
                BuiltinEngines.scalar.axpy(
                    expected,
                    destination.offset,
                    alpha,
                    source.toDoubleArray(),
                    0,
                    source.size,
                    destination.stride,
                )

                destination.axpy(alpha, source)

                assertAxpyAgreesWithReference(expected, source.values)
            }
        }
    }

    @Test
    fun `identical mapping copy preserves the full backing buffer`() {
        for ((source, destination) in identicalDenseMappings()) {
            val expected = source.values.copyOf()

            copy(source, destination)

            assertContentEquals(expected, source.values)
        }
    }

    @Test
    fun `identical mapping swap preserves the full backing buffer`() {
        for ((source, destination) in identicalDenseMappings()) {
            val expected = source.values.copyOf()

            swap(source, destination)

            assertContentEquals(expected, source.values)
        }
    }

    private fun identicalDenseMappings(): List<Pair<DenseVector, DenseVector>> = buildList {
        for (size in intArrayOf(0, 1, 5, 129)) {
            for (sameWrapper in booleanArrayOf(false, true)) {
                val whole = DenseVector.wrap(DoubleArray(size) { it + 0.25 })
                add(whole to if (sameWrapper) whole else whole.asView())
                for (stride in intArrayOf(1, 3, -1, -3)) {
                    val backing = DoubleArray(3 * size + 4) { it + 0.25 }
                    val offset = if (stride > 0 || size == 0) 2 else 2 + (size - 1) * -stride
                    val source = StridedVector(backing, offset, size, stride)
                    add(source to if (sameWrapper) source else source.asView())
                }
            }
        }
    }

    @Test
    fun `reversed overlapping axpy agrees with the scalar snapshot reference`() {
        val backing = DoubleArray(12) { it + 0.25 }
        val source = StridedVector(backing, 2, 5)
        val destination = StridedVector(backing, 6, 5, -1)
        val expected = backing.copyOf()
        BuiltinEngines.scalar.axpy(expected, 6, 0.75, source.toDoubleArray(), 0, 5, -1)

        destination.axpy(0.75, source)

        assertAxpyAgreesWithReference(expected, backing)
    }

    @Test
    fun `sparse dot against dense views agrees with the scalar reference`() {
        val source = sparse(5, 0 to 2.0, 2 to -1.5, 4 to 0.0)
        for (stride in intArrayOf(-3, -1, 1, 3)) {
            val backing = DoubleArray(18) { it * 0.125 - 0.75 }
            val view = StridedVector(backing, if (stride > 0) 2 else 14, 5, stride)
            val expected = BuiltinEngines.scalar.dot(source, view.toDoubleArray())

            assertDotAgreesWithReference(expected, source dot view)
            assertDotAgreesWithReference(expected, view dot source)
        }
    }

    @Test
    fun `sparse dot against dense views evaluates stored zeros`() {
        val source = sparse(3, 1 to 0.0)
        val view = StridedVector(doubleArrayOf(Double.NaN, 1.0, Double.POSITIVE_INFINITY, 1.0, Double.NaN), 0, 3, 2)
        val expected = BuiltinEngines.scalar.dot(source, view.toDoubleArray())

        assertDotAgreesWithReference(expected, source dot view)
        assertDotAgreesWithReference(expected, view dot source)
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double) {
        if (expected.isNaN()) assertTrue(actual.isNaN()) else assertEquals(expected, actual, 1e-12)
    }

    /** A non-koblas vector storage, which reaches the public adapter fallbacks rather than a kernel. */
    private class ForeignVector(private val backing: DenseVector) : Vector {
        override val size: Int get() = backing.size
        override fun get(i: Int): Double = backing[i]
        override fun toDoubleArray(): DoubleArray = backing.toDoubleArray()
    }

    private fun foreignRamp(size: Int) = ForeignVector(DenseVector.of(DoubleArray(size) { it * 0.5 - 1.0 }))

    private val sparse = SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    private fun dense(vararg values: Double) = DenseVector.of(values)
    private fun sparse(size: Int, vararg entries: Pair<Int, Double>) = SparseVector.of(
        size,
        entries.map { it.first }.toIntArray(),
        entries.map { it.second }.toDoubleArray(),
    )

    @Test
    fun `sparse axpy accepts a prefix view and preserves its padding`() {
        for (size in intArrayOf(0, 3)) {
            val backing = DoubleArray(7) { it + 0.5 }
            val destination = StridedVector(backing, 0, size)
            val source = if (size == 0) sparse(0) else sparse(size, 0 to 2.0, 2 to -1.0)
            val expected = backing.copyOf()
            BuiltinEngines.scalar.axpy(source.indices, 0, source.values, 0, source.indices.size, 2.0, expected)

            destination.axpy(2.0, source)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    private fun assertAxpyAgreesWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            when {
                expected[i].isNaN() -> assertTrue(actual[i].isNaN(), "axpy backing index $i should be NaN")
                expected[i] == 0.0 -> assertEquals(expected[i].toBits(), actual[i].toBits(), "axpy backing index $i")
                else -> assertClose(expected[i], actual[i], "axpy backing index $i")
            }
        }
    }

    @Test
    fun `dot is symmetric and sparsity agnostic`() {
        val a = dense(1.0, 2.0, 3.0, 4.0)
        val b = dense(0.5, 0.0, -1.0, 2.0)
        val bSparse = sparse(4, 0 to 0.5, 2 to -1.0, 3 to 2.0)
        val expected = BuiltinEngines.scalar.dot(a.values, 0, b.values, 0, 4)
        val sparseByDense = BuiltinEngines.scalar.dot(b.values, 0, b.values, 0, 4)

        for ((actual, context) in listOf(
            (a dot b) to "dense by dense",
            (b dot a) to "dense by dense reversed",
            (a dot bSparse) to "dense by sparse",
            (bSparse dot a) to "sparse by dense",
        )) {
            assertReductionAgreesWithReference(expected, actual, context)
        }
        assertReductionAgreesWithReference(sparseByDense, bSparse dot b, "sparse by its own dense form")
    }

    @Test
    fun `axpy adds alpha scaled x to y for any sparsity`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        y.axpy(2.0, sparse(3, 0 to 1.0, 2 to -1.0))
        assertEquals(dense(3.0, 2.0, 1.0), y)
    }

    @Test
    fun `scale mutates in place and respects identity`() {
        val v = DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        v.scale(0.5)
        assertEquals(dense(0.5, -1.0, 1.5), v)
        v.scale(1.0)
        assertEquals(dense(0.5, -1.0, 1.5), v)
    }

    @Test
    fun `norm two agrees with the scalar oracle on dense and sparse`() {
        val values = doubleArrayOf(3.0, 0.0, -4.0)
        assertReductionAgreesWithReference(
            BuiltinEngines.scalar.nrm2(values, 0, values.size),
            DenseVector.of(values).norm2(),
            "dense",
        )
        assertReductionAgreesWithReference(
            BuiltinEngines.scalar.nrm2(denseOfSparse.values, 0, denseOfSparse.size),
            sparse.norm2(),
            "sparse",
        )
        assertEquals(0.0, DenseVector.zero(4).norm2())
        assertEquals(0.0, DenseVector.zero(0).norm2())
    }

    @Test
    fun `norm two survives overflow and underflow via the rescale fallback`() {
        assertEquals(5.0e200, DenseVector.of(doubleArrayOf(3.0e200, 0.0, -4.0e200)).norm2(), 1e186)
        assertEquals(5.0e-200, DenseVector.of(doubleArrayOf(3.0e-200, 4.0e-200)).norm2(), 1e-214)
        assertEquals(1.0e-300, DenseVector.of(doubleArrayOf(1.0e-300)).norm2(), 1e-314)
        val sparseHuge = SparseVector.of(5, intArrayOf(0, 3), doubleArrayOf(3.0e200, 4.0e200))
        assertEquals(5.0e200, sparseHuge.norm2(), 1e186)
    }

    @Test
    fun `sum adds the entries for any storage`() {
        val values = doubleArrayOf(2.0, -3.5, 0.0, 4.25, -1.0, 0.5)
        val expected = BuiltinEngines.scalar.sum(values, 0, values.size)
        val strided = StridedVector(DoubleArray(12) { values[it / 2] }, 0, 6, 2)
        assertReductionAgreesWithReference(expected, DenseVector.of(values).sum(), "dense")
        assertReductionAgreesWithReference(expected, strided.sum(), "strided")
        assertReductionAgreesWithReference(
            BuiltinEngines.scalar.sum(denseOfSparse.values, 0, denseOfSparse.size),
            sparse.sum(),
            "sparse",
        )
        assertEquals(0.0, DenseVector.of(DoubleArray(0)).sum(), "empty")
    }

    @Test
    fun `asum agrees with the scalar oracle on dense and sparse`() {
        val values = doubleArrayOf(3.0, 0.0, -4.0)
        assertReductionAgreesWithReference(
            BuiltinEngines.scalar.asum(values, 0, values.size),
            DenseVector.of(values).asum(),
            "dense",
        )
        assertReductionAgreesWithReference(
            BuiltinEngines.scalar.asum(denseOfSparse.values, 0, denseOfSparse.size),
            sparse.asum(),
            "sparse",
        )
        assertEquals(0.0, DenseVector.zero(0).asum())
    }

    @Test
    fun `iamax returns a maximal index and handles edge cases`() {
        assertEquals(2, DenseVector.of(doubleArrayOf(1.0, -2.0, 5.0, -4.0)).iamax())
        assertEquals(4, sparse.iamax())
        assertEquals(0, DenseVector.zero(3).iamax()) // zero vector: first element
        assertEquals(0, SparseVector.of(3, IntArray(0), DoubleArray(0)).iamax()) // all-unstored: same
        assertEquals(-1, DenseVector.zero(0).iamax())
    }

    @Test
    fun `public iamax returns logical indices across strides and kernel boundaries`() {
        // A unit stride reaches the kernel unchanged, and the engine tests already hold it to this contract.
        for (stride in intArrayOf(2, -2)) {
            val publicKernels = object : EngineKernels {
                override val implementations = setOf(EngineImplementation.Iamax)
                override val name: String = "test-VectorOpsTest"

                override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
                    val values = DoubleArray(len) { v[vOff + it * vStride] }
                    val step = if (stride > 0) stride else -stride
                    val offset = 1 + if (stride < 0) (len - 1).coerceAtLeast(0) * step else 0
                    val backing = DoubleArray(2 + len * step) { Double.POSITIVE_INFINITY }
                    for (i in 0 until len) backing[offset + i * stride] = values[i]
                    return StridedVector(backing, offset, len, stride).iamax()
                }
            }

            assertIamaxHonoursItsContract(publicKernels)
        }
    }

    @Test
    fun `iamax reads a stored zero as the zero it is`() {
        assertEquals(0, SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(0, SparseVector.of(5, intArrayOf(1, 3), doubleArrayOf(0.0, -0.0)).iamax())
        assertEquals(DenseVector.zero(5).iamax(), SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(4, SparseVector.of(5, intArrayOf(1, 4), doubleArrayOf(0.0, -2.0)).iamax())
    }

    @Test
    fun `copy replicates dense and sparse sources and rejects size mismatch`() {
        val dst = DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0, 9.0, 9.0, 9.0))
        copy(sparse, dst) // sparse: must zero-fill the unstored slots
        assertContentEquals(denseOfSparse.values, dst.values)
        copy(DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), dst)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dst.values)
        assertFailsWith<DimensionMismatch> { copy(DenseVector.zero(2), DenseVector.zero(3)) }
    }

    @Test
    fun `swap exchanges contents and rejects size mismatch`() {
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(3.0, 4.0))
        swap(a, b)
        assertContentEquals(doubleArrayOf(3.0, 4.0), a.values)
        assertContentEquals(doubleArrayOf(1.0, 2.0), b.values)
        assertFailsWith<DimensionMismatch> { swap(DenseVector.zero(2), DenseVector.zero(3)) }
    }

    @Test
    fun `vector operations reject mismatched sizes`() {
        assertFailsWith<DimensionMismatch> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<DimensionMismatch> { dense(1.0).axpy(1.0, dense(1.0, 2.0)) }
    }

    @Test
    fun `sparse against sparse dot matches the dense answer over merge shapes`() {
        val n = 8
        val patterns = listOf(
            intArrayOf(0, 2, 4, 6) to intArrayOf(1, 3, 5, 7),
            intArrayOf(0, 1, 2) to intArrayOf(0, 1, 2),
            intArrayOf(0, 7) to intArrayOf(3, 4),
            intArrayOf(0, 1, 2, 3) to intArrayOf(3),
            intArrayOf(5) to intArrayOf(0, 1, 2, 3, 4),
            IntArray(0) to intArrayOf(0, 4),
            IntArray(0) to IntArray(0),
        )
        for ((ia, ib) in patterns) {
            val a = SparseVector.of(n, ia, DoubleArray(ia.size) { it + 1.5 })
            val b = SparseVector.of(n, ib, DoubleArray(ib.size) { it + 2.5 })
            val expected = DenseVector.of(a.toDoubleArray()) dot DenseVector.of(b.toDoubleArray())
            assertEquals(expected, a dot b, "pattern ${ia.toList()} vs ${ib.toList()}")
            assertEquals(expected, b dot a, "dot should be symmetric for ${ia.toList()} vs ${ib.toList()}")
        }
    }

    @Test
    fun `sparse dot ignores unstored entries in borrowed and foreign vectors`() {
        val sparse = sparse(3, 1 to 2.0)
        val backing = doubleArrayOf(Double.NaN, 3.0, Double.POSITIVE_INFINITY)
        val borrowed = StridedVector(backing, 2, 3, -1)
        val foreign = ForeignVector(DenseVector.wrap(backing))

        for (other in listOf(borrowed, foreign)) {
            val expected = sparse dot DenseVector.of(other.toDoubleArray())

            assertEquals(expected, sparse dot other)
            assertEquals(expected, other dot sparse)
        }
    }

    @Test
    fun `copy snapshots a foreign source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = if (borrowed) {
                doubleArrayOf(9.0, 1.0, 2.0, 3.0, 4.0, 9.0)
            } else {
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            }
            val destination = if (borrowed) StridedVector(backing, 1, 4) else DenseVector.wrap(backing)
            val source = ForeignVector(StridedVector(backing, destination.offset + 3, 4, -1))
            val expected = backing.copyOf()
            val snapshot = source.toDoubleArray()
            snapshot.copyInto(expected, destination.offset)

            copy(source, destination)

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `axpy snapshots a foreign source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = if (borrowed) {
                doubleArrayOf(9.0, 1.0, 2.0, 3.0, 4.0, 9.0)
            } else {
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            }
            val destination = if (borrowed) StridedVector(backing, 1, 4) else DenseVector.wrap(backing)
            val source = ForeignVector(StridedVector(backing, destination.offset + 3, 4, -1))
            val expected = backing.copyOf()
            BuiltinEngines.scalar.axpy(expected, destination.offset, 2.0, source.toDoubleArray(), 0, 4)

            destination.axpy(2.0, source)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `copy into dense storage preserves a borrowed source sharing its buffer`() {
        for (reverse in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = StridedVector(backing, if (reverse) 3 else 0, 4, if (reverse) -1 else 1)
            val expected = source.toDoubleArray()

            copy(source, DenseVector.wrap(backing))

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `copy preserves sparse values shared with the destination`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0)
            val source = SparseVector.wrap(3, intArrayOf(0, 1, 2), backing)
            val expected = if (borrowed) backing.reversedArray() else backing.copyOf()

            if (borrowed) {
                copy(source, StridedVector(backing, 2, 3, -1))
            } else {
                copy(source, DenseVector.wrap(backing))
            }

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `axpy preserves a borrowed source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = StridedVector(backing, 3, 4, -1)
            val expected = DoubleArray(backing.size) { backing[it] + 2.0 * source[it] }

            if (borrowed) {
                StridedVector(backing, 0, 4).axpy(2.0, source)
            } else {
                DenseVector.wrap(backing).axpy(2.0, source)
            }

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `borrowed axpy with sparse input leaves unstored positions untouched`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0)
        val source = sparse(3, 1 to 2.0)
        val destination = StridedVector(backing, 2, 3, -1)
        val expected = DenseVector.of(destination.toDoubleArray())
        expected.axpy(Double.POSITIVE_INFINITY, source)

        destination.axpy(Double.POSITIVE_INFINITY, source)

        assertContentEquals(expected.values.reversedArray(), backing)
    }

    private fun assertReductionAgreesWithReference(expected: Double, actual: Double, context: String) {
        assertClose(expected, actual, context)
    }

    @Test
    fun `the norm accepts a foreign vector`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val expected = PortableEngine.nrm2(values, 0, values.size)

        val actual = source.norm2()

        assertReductionAgreesWithReference(expected, actual, "foreign norm")
    }

    @Test
    fun `the absolute sum accepts a foreign vector`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val expected = PortableEngine.asum(values, 0, values.size)

        val actual = source.asum()

        assertReductionAgreesWithReference(expected, actual, "foreign asum")
    }

    @Test
    fun `the index search accepts a foreign vector`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val expected = PortableEngine.iamax(values, 0, values.size)

        val actual = source.iamax()

        assertEquals(expected, actual)
    }

    @Test
    fun `dot accepts a foreign vector on either side`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val dense = DenseVector.of(values)
        val expected = PortableEngine.dot(values, 0, values, 0, values.size)

        for ((left, right) in listOf(source to source, dense to source, source to dense)) {
            val actual = left dot right

            assertReductionAgreesWithReference(expected, actual, "foreign dot")
        }
    }

    @Test
    fun `axpy accepts a foreign source`() {
        val source = foreignRamp(6)
        val destination = DenseVector.zero(6)
        val expected = DoubleArray(6) { 2.0 * source[it] }

        destination.axpy(2.0, source)

        assertClose(expected, destination.values, "foreign axpy")
    }

    @Test
    fun `copy accepts a foreign source`() {
        val source = foreignRamp(6)
        val destination = DenseVector.zero(6)

        copy(source, destination)

        assertContentEquals(source.toDoubleArray(), destination.values)
    }

    @Test
    fun `dot reads the logical entries of borrowed slices`() {
        val backing = DoubleArray(8) { it + 1.0 }
        val even = StridedVector(backing, 0, 4, 2)
        val odd = StridedVector(backing, 1, 4, 2)
        val expected = PortableEngine.dot(backing, 0, backing, 1, 4, 2, 2)

        val actual = even dot odd

        assertClose(expected, actual, "borrowed dot")
    }

    @Test
    fun `scaling a borrowed slice leaves other entries untouched`() {
        val backing = DoubleArray(8) { it + 1.0 }
        val even = StridedVector(backing, 0, 4, 2)
        val expected = backing.copyOf()
        PortableEngine.scale(expected, 0, 2.0, 4, 2)

        even.scale(2.0)

        assertContentEquals(expected, backing)
    }

    @Test
    fun `alpha zero leaves the destination unchanged`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))

        y.axpy(0.0, DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0)))

        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0), y.values)
    }

    /** What the routine is for: a rotation built from the leading pair eliminates the second of them. */
    @Test
    fun `a rotation from the leading pair zeroes the target entry`() {
        // `(0.6, 0.8)` is the rotation `drotg(3, 4)` produces.
        val x = DenseVector.of(doubleArrayOf(3.0, 1.0, 2.0))
        val y = DenseVector.of(doubleArrayOf(4.0, -1.0, 0.5))

        rot(x, y, 0.6, 0.8)

        assertEquals(0.0, y[0], 1e-12, "the leading entry of y was not eliminated")
        assertEquals(5.0, x[0], 1e-12)
    }

    // `rot` addresses each operand through its own offset and stride, so one that ignored them would rotate the
    // padding and report nothing. The run vectorises everywhere, and that kernel fuses its multiply and add,
    // so only the padding is compared exactly.
    @Test
    fun `a rotation over strided operands touches only the entries they address`() {
        val rng = Random(20260916)
        val n = 8
        val pad = 2
        val xs = randomVector(pad + n + pad, rng)
        val ys = randomVector(pad + n + pad, rng)
        val x0 = xs.copyOf()
        val y0 = ys.copyOf()
        val expectedX = xs.copyOf()
        val expectedY = ys.copyOf()
        PortableEngine.rot(expectedX, pad, expectedY, pad, n, 0.6, 0.8)

        rot(StridedVector(xs, pad, n), StridedVector(ys, pad, n), c = 0.6, s = 0.8)

        assertClose(expectedX, xs, "rotation x backing")
        assertClose(expectedY, ys, "rotation y backing")
        for (i in listOf(0, 1, pad + n, pad + n + 1)) {
            assertEquals(x0[i], xs[i], "x padding at $i")
            assertEquals(y0[i], ys[i], "y padding at $i")
        }
    }

    @Test
    fun `gather rejects a dense vector of another length`() {
        val pattern = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(9.0, 9.0))

        assertFailsWith<DimensionMismatch> { gather(pattern, DenseVector.of(DoubleArray(7))) }
    }

    @Test
    fun `gatherZero rejects a dense vector of another length`() {
        val pattern = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(9.0, 9.0))

        assertFailsWith<DimensionMismatch> { gatherZero(pattern, DenseVector.of(DoubleArray(7))) }
    }
}
