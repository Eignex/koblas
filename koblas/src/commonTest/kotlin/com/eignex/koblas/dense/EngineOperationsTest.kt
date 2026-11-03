package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.sparse.*
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class EngineOperationsTest {
    @Test
    fun `platform scaling agrees with reference across windows`() = assertScaleAgreesWithReference(koblas)

    @Test
    fun `simd scaling agrees with reference across windows`() {
        val simd = BuiltinEngines.simd ?: return
        assertScaleAgreesWithReference(simd)
    }

    @Test
    fun `the compiled in kernels name themselves for engine attribution`() {
        assertTrue(koblas.name.isNotEmpty())
    }

    @Test
    fun `the compiled in kernels keep the public scale and axpy noops`() {
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        koblas.axpy(y, 0, 0.0, x, 0, 64)
        koblas.scale(x, 0, 1.0, 64)

        assertTrue(y.all { it == 0.0 }, "zero axpy must not evaluate infinity times zero")
        assertTrue(x.all { it == Double.POSITIVE_INFINITY }, "unit scale changed the vector")
    }

    @Test
    fun `the scalar norm rescales components whose squares exceed the exponent range`() {
        for (scale in doubleArrayOf(1e200, 1e-200)) {
            val input = doubleArrayOf(3.0 * scale, 4.0 * scale)

            val actual = PortableEngine.nrm2(input, 0, input.size)

            assertEquals(5.0 * scale, actual, absoluteTolerance = scale * 1e-12)
        }
    }

    @Test
    fun `the scalar norm preserves equal components across lengths`() {
        for (length in intArrayOf(16, 33, 64)) {
            for (scale in doubleArrayOf(1e200, 1e-200)) {
                val input = DoubleArray(length) { scale }
                val expected = sqrt(length.toDouble()) * scale

                val actual = PortableEngine.nrm2(input, 0, length)

                assertEquals(expected, actual, absoluteTolerance = expected * 1e-12)
            }
        }
    }

    @Test
    fun `norm agrees with the scalar oracle at extreme magnitudes`() = assertNormExtremesAgreeWithReference(koblas)

    @Test
    fun `norm agrees with the scalar oracle across offsets and lengths`() = assertNormWindowsAgreeWithReference(koblas)

    /**
     * A caller walking a matrix reaches its last row or column with an empty tail, so every routine here is
     * called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every kernel accepts a zero length run`() {
        val vectorKernels = koblas
        val v = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, vectorKernels.dot(v, 3, v, 3, 0), "dot over nothing")
        assertEquals(0.0, vectorKernels.nrm2(v, 3, 0), "nrm2 over nothing")
        assertEquals(0.0, vectorKernels.asum(v, 3, 0), "asum over nothing")
        assertEquals(0.0, vectorKernels.sum(v, 3, 0), "sum over nothing")
        assertEquals(-1, vectorKernels.iamax(v, 3, 0), "iamax over nothing")
        vectorKernels.axpy(v, 3, 2.0, v, 3, 0)
        vectorKernels.scale(v, 3, 2.0, 0)
        vectorKernels.swap(v, 3, v, 3, 0)
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
    }

    @Test
    fun `the compiled in level one kernels agree with the scalar loops`() =
        assertLevel1KernelsAgreeWithReference(koblas)

    @Test
    fun `the compiled in iamax honours its contract`() = assertIamaxHonoursItsContract(koblas)

    @Test
    fun `simd iamax honours its contract across windows`() {
        val simd = BuiltinEngines.simd ?: return
        assertIamaxHonoursItsContract(simd)
    }

    @Test
    fun `the compiled in reductions agree with the scalar loops`() = assertReductionsAgreeWithReference(koblas)

    @Test
    fun `the compiled in swap agrees with the scalar loop`() = assertSwapAgreesWithReference(koblas)

    @Test
    fun `the compiled in rot kernel agrees with the portable one`() = assertRotKernelAgreesWithReference(koblas)

    private val eps = 2.220446049250313e-16

    private fun infNorm(v: DoubleArray): Double {
        var m = 0.0
        for (x in v) m = maxOf(m, abs(x))
        return m
    }

    private fun infNorm(a: DenseMatrix): Double {
        var m = 0.0
        for (i in 0 until a.rows) {
            var r = 0.0
            for (j in 0 until a.cols) r += abs(a[i, j])
            m = maxOf(m, r)
        }
        return m
    }

    @Test
    fun `full gemv agrees with the scalar oracle across alpha beta and transpose`() = withEngines { blas ->
        val rng = Random(20260727)
        for (n in intArrayOf(1, 4, 17)) {
            val m = n + 3 // non-square to catch row/col mixups
            val a = DenseMatrix(m, n, DoubleArray(m * n) { rng.nextDouble(-1.0, 1.0) })
            for (transpose in booleanArrayOf(false, true)) {
                val xLen = if (transpose) m else n
                val yLen = if (transpose) n else m
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        val x = DoubleArray(xLen) { rng.nextDouble(-2.0, 2.0) }
                        // beta == 0 must overwrite without reading, so y starts poisoned with NaN.
                        val y0 = DoubleArray(yLen) { if (beta == 0.0) Double.NaN else rng.nextDouble(-2.0, 2.0) }
                        val expected = y0.copyOf()
                        ReferenceBlas.gemv(alpha, a, x, beta, expected, transpose)
                        val y = y0.copyOf()
                        blas.gemv(alpha, a, x, beta, y, transpose)
                        val bound = 100.0 * maxOf(m, n) * eps * (infNorm(a) * infNorm(x) + infNorm(expected))
                        for (i in 0 until yLen) {
                            assertTrue(
                                abs(y[i] - expected[i]) <= bound + 1e-12,
                                "gemv n=$n t=$transpose a=$alpha b=$beta at $i: ${y[i]} vs ${expected[i]}",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `gemv and gemm reject incompatible shapes`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix(2, 3) * DenseMatrix(2, 2) }
        assertFailsWith<DimensionMismatch> { koblas.gemv(DenseMatrix(2, 3), DoubleArray(2)) }
        assertFailsWith<DimensionMismatch> {
            koblas.gemv(DenseMatrix(2, 3), DoubleArray(3), transpose = true)
        }
    }

    @Test
    fun `allocating gemm rejects inner dimensions before result capacity`() = withEngines { blas ->
        val a = DenseMatrix.zero(Int.MAX_VALUE, 0)
        val b = DenseMatrix.zero(1, 2)

        val failure = assertFailsWith<DimensionMismatch> { blas.gemm(a, b) }

        assertEquals("gemm: inner dimensions differ, 0 vs 1", failure.message)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a tall result in every mode`() = withEngines { blas ->
        checkGemmShape(blas, Random(20260728), m = 5, k = 7, n = 4)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a wide result in every mode`() = withEngines { blas ->
        checkGemmShape(blas, Random(20260728), m = 3, k = 7, n = 6)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a single row result in every mode`() = withEngines { blas ->
        val rng = Random(20261101)
        checkGemmShape(blas, rng, m = 1, k = 7, n = 6)
        checkGemmShape(blas, rng, m = 1, k = 1, n = 9)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a single column result in every mode`() = withEngines { blas ->
        val rng = Random(20261102)
        checkGemmShape(blas, rng, m = 6, k = 7, n = 1)
        checkGemmShape(blas, rng, m = 9, k = 1, n = 1)
    }

    @Test
    fun `gemm agrees with the scalar oracle for a single entry result in every mode`() = withEngines { blas ->
        checkGemmShape(blas, Random(20261103), m = 1, k = 7, n = 1)
    }

    private fun checkGemmShape(blas: KoblasEngine, rng: Random, m: Int, k: Int, n: Int) {
        for (tA in booleanArrayOf(false, true)) {
            for (tB in booleanArrayOf(false, true)) {
                val a = if (tA) DenseMatrix(k, m) else DenseMatrix(m, k)
                val b = if (tB) DenseMatrix(n, k) else DenseMatrix(k, n)
                for (idx in a.values.indices) a.values[idx] = rng.nextDouble(-1.0, 1.0)
                for (idx in b.values.indices) b.values[idx] = rng.nextDouble(-1.0, 1.0)
                for (alpha in doubleArrayOf(0.0, 1.0, -2.0)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        // beta == 0 must overwrite without reading, so C starts poisoned with NaN.
                        val c0 = DenseMatrix(m, n)
                        for (idx in c0.values.indices) {
                            c0.values[idx] = if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                        }
                        val expected = DenseMatrix(m, n, c0.values.copyOf())
                        ReferenceBlas.gemm(alpha, a, tA, b, tB, beta, expected)
                        val c = DenseMatrix(m, n, c0.values.copyOf())
                        blas.gemm(alpha, a, tA, b, tB, beta, c)
                        val bound = 100.0 * k * eps * (infNorm(a) * infNorm(b) + infNorm(expected))
                        for (idx in c.values.indices) {
                            assertTrue(
                                abs(c.values[idx] - expected.values[idx]) <= bound + 1e-12,
                                "gemm tA=$tA tB=$tB a=$alpha b=$beta at $idx: " +
                                    "${c.values[idx]} vs ${expected.values[idx]}",
                            )
                        }
                    }
                }
            }
        }
    }

    private fun engines() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()

    @Test
    fun `windowed GER agrees with the reference with independent offsets and parent strides`() {
        for (engine in engines()) {
            for ((rows, columns) in listOf(1 to 1, 3 to 2, 5 to 4)) {
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                val x = DoubleArray(10) { (it - 4) / 16.0 }
                val y = DoubleArray(11) { (5 - it) / 8.0 }
                val originalX = x.copyOf()
                val originalY = y.copyOf()
                val expected = referenceWindow(a, x, y, rows, columns)

                engine.ger(-0.875, x, y, a, 2, 3, rows, columns, 1, 4)

                assertWindowAgreesWithReference(expected, a, rows, columns)
                assertContentEquals(originalX, x)
                assertContentEquals(originalY, y)
            }
        }
    }

    @Test
    fun `windowed GER snapshots either or both aliased vector ranges`() {
        for (engine in engines()) {
            for (alias in 1..3) {
                val workspace = Workspace()
                repeat(2) { _ ->
                    val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                    val x = if (alias != 2) a.values else DoubleArray(10) { (it - 4) / 16.0 }
                    val y = if (alias != 1) a.values else DoubleArray(11) { (5 - it) / 8.0 }
                    val xOffset = if (alias != 2) 29 else 1
                    val yOffset = if (alias != 1) 30 else 4
                    val expected = referenceWindow(a, x, y, 3, 2, xOffset, yOffset)

                    engine.ger(-0.875, x, y, a, 2, 3, 3, 2, xOffset, yOffset, workspace)

                    assertWindowAgreesWithReference(expected, a, 3, 2)
                    assertEquals(if (alias != 2) 1 else 0, workspace.available(3))
                    assertEquals(if (alias != 1) 1 else 0, workspace.available(2))
                    assertEquals(0, workspace.available(a.values.size))
                }
            }
        }
    }

    @Test
    fun `windowed GER validates every range before writing even with zero alpha`() {
        val valid = intArrayOf(2, 3, 3, 2, 1, 4)
        for (engine in engines()) {
            for (alpha in doubleArrayOf(0.0, -1.0)) {
                for (index in valid.indices) {
                    for (invalid in intArrayOf(-1, Int.MAX_VALUE)) {
                        val args = valid.copyOf().also { it[index] = invalid }
                        val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { it.toDouble() })
                        val before = a.values.copyOf()

                        assertFailsWith<DimensionMismatch> {
                            engine.ger(
                                alpha, DoubleArray(10), DoubleArray(11), a,
                                args[0], args[1], args[2], args[3], args[4], args[5],
                            )
                        }

                        assertContentEquals(before, a.values)
                    }
                }
            }
        }
    }

    @Test
    fun `empty windows and zero alpha do not read entries or borrow scratch`() {
        for (engine in engines()) {
            for (shape in listOf(0 to 0, 0 to 2, 3 to 0, 3 to 2)) {
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { Double.NaN })
                val workspace = Workspace()
                val alpha = if (shape.first > 0 && shape.second > 0) 0.0 else Double.NaN

                engine.ger(
                    alpha, a.values, a.values, a,
                    9 - shape.first, 8 - shape.second, shape.first, shape.second,
                    72 - shape.first, 72 - shape.second, workspace,
                )

                assertContentEquals(DoubleArray(72) { Double.NaN }, a.values)
                assertEquals(0, workspace.idleLengths())
            }
        }
    }

    @Test
    fun `windowed GER agrees with the reference on shrinking supports and extreme finite inputs`() {
        for (engine in engines()) {
            assertShrinkingGerAgreesWithReference { x, y, a ->
                val start = x.indexOfFirst { it != 0.0 }
                val extent = a.rows - start
                engine.ger(-1.0, x, y, a, start, start, extent, extent, start, start)
            }
        }
    }

    @Suppress("LongParameterList") // the rectangle and independently selected input ranges
    private fun referenceWindow(
        a: DenseMatrix,
        x: DoubleArray,
        y: DoubleArray,
        rows: Int,
        columns: Int,
        xOffset: Int = 1,
        yOffset: Int = 4,
    ): DenseMatrix {
        val expected = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())
        val rectangle = DenseMatrix.wrap(
            rows,
            columns,
            DoubleArray(rows * columns) { index ->
                a[2 + index % rows, 3 + index / rows]
            },
        )
        ReferenceBlas.ger(
            -0.875,
            x.copyOfRange(xOffset, xOffset + rows),
            y.copyOfRange(yOffset, yOffset + columns),
            rectangle,
        )
        for (j in 0 until columns) for (i in 0 until rows) expected[2 + i, 3 + j] = rectangle[i, j]
        return expected
    }

    private fun assertWindowAgreesWithReference(expected: DenseMatrix, actual: DenseMatrix, rows: Int, columns: Int) {
        assertClose(expected, actual, "windowed GER")
        for (j in 0 until actual.cols) {
            for (i in 0 until actual.rows) {
                if (i !in 2 until 2 + rows || j !in 3 until 3 + columns) {
                    assertEquals(expected[i, j], actual[i, j], "outside rectangle at $i $j")
                }
            }
        }
    }
}
