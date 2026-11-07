@file:Suppress("LongParameterList") // a product case is its shape, its two transposes and its operand layout

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.operations.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DenseOpsTest {
    // A destination of one row or one column is a matrix-vector product, which reads the vector operand along a
    // stored row as often as along a stored column, so the padding of a view decides the stride it is read at.
    @Test
    fun `gemm agrees with the oracle at every shape transpose and scaling on contiguous and view operands`() =
        withRegionEngines { blas ->
            for ((m, k, n) in GEMM_SHAPES) {
                for (transposeA in booleanArrayOf(false, true)) {
                    for (transposeB in booleanArrayOf(false, true)) {
                        for (view in booleanArrayOf(false, true)) {
                            assertGemmAgreesWithReference(blas, m, k, n, transposeA, transposeB, view)
                        }
                    }
                }
            }
        }

    private fun assertGemmAgreesWithReference(
        blas: KoblasEngine,
        m: Int,
        k: Int,
        n: Int,
        transposeA: Boolean,
        transposeB: Boolean,
        view: Boolean,
    ) {
        val a = operand(if (transposeA) k else m, if (transposeA) m else k, 1, view)
        val b = operand(if (transposeB) n else k, if (transposeB) k else n, 2, view)
        for (alpha in doubleArrayOf(0.0, 1.0, -2.0)) {
            for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                val c = operand(m, n, 3, view)
                // A zero beta overwrites without reading, so the destination starts poisoned.
                if (beta == 0.0) for (j in 0 until n) for (i in 0 until m) c[i, j] = Double.NaN
                val before = c.values.copyOf()
                val expected = contiguousCopy(c)
                ReferenceBlas.gemm(alpha, contiguousCopy(a), transposeA, contiguousCopy(b), transposeB, beta, expected)

                blas.gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch = Scratch())

                val context = "${blas.name} ${m}x${k}x$n tA=$transposeA tB=$transposeB view=$view a=$alpha b=$beta"
                assertRegionAgreesWithReference(expected, c, before, context)
            }
        }
    }

    private fun operand(rows: Int, cols: Int, seed: Int, view: Boolean): DenseMatrix {
        val region = paddedRegion(rows, cols, seed)
        return if (view) region else contiguousCopy(region)
    }

    @Test
    fun `packed products retain destination padding`() = withRegionEngines { blas ->
        val a = paddedRegion(67, 65)
        val b = paddedRegion(65, 69, 2)
        val c = paddedRegion(67, 69, 3)
        val expected = contiguousCopy(c)
        val before = c.values.copyOf()
        ReferenceBlas.gemm(1.0, contiguousCopy(a), false, contiguousCopy(b), false, 0.0, expected)

        blas.gemm(1.0, a, false, b, false, 0.0, c, scratch = Scratch())

        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `full gemv agrees with the scalar oracle across alpha beta and transpose`() = withEngines { blas ->
        val rng = Random(20260727)
        for (n in intArrayOf(1, 4, 17)) {
            // Not square, so a row and column mixup cannot pass.
            val m = n + 3
            val a = randomMatrix(m, n, rng)
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 1.0, -1.5)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        val x = DoubleArray(if (transpose) m else n) { rng.nextDouble(-2.0, 2.0) }
                        // A zero beta overwrites without reading, so the output starts poisoned.
                        val start = DoubleArray(if (transpose) n else m) {
                            if (beta == 0.0) Double.NaN else rng.nextDouble(-2.0, 2.0)
                        }
                        val expected = start.copyOf()
                        ReferenceBlas.gemv(alpha, a, x, beta, expected, transpose)
                        val actual = start.copyOf()

                        blas.kernels.gemv(alpha, a, x, 0, 1, beta, actual, 0, 1, transpose)

                        assertClose(expected, actual, "gemv n=$n t=$transpose a=$alpha b=$beta")
                    }
                }
            }
        }
    }

    @Test
    fun `matrix vector calls address the borrowed rectangle`() = withRegionEngines { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            val a = paddedRegion(7, 5)
            val x = DoubleArray(if (transpose) 7 else 5) { it / 8.0 }
            val actual = DoubleArray(if (transpose) 5 else 7) { it / 16.0 }
            val expected = actual.copyOf()
            ReferenceBlas.gemv(0.875, contiguousCopy(a), x, -0.25, expected, transpose)

            blas.kernels.gemv(0.875, a, x, 0, 1, -0.25, actual, 0, 1, transpose)

            assertClose(expected, actual, "view vector")
        }
    }

    @Test
    fun `matrix vector routines address positive and negative strides`() = withRegionEngines { blas ->
        val a = paddedRegion(5, 5)
        for (stride in intArrayOf(2, -2)) {
            for (transpose in booleanArrayOf(false, true)) {
                val backing = DoubleArray(17) { (it - 8) / 7.0 }
                val x = DenseVector.wrap(backing, if (stride > 0) 2 else 10, 5, stride)
                val output = DoubleArray(19) { -77.0 }
                val y = DenseVector.wrap(output, if (stride > 0) 3 else 11, 5, stride)
                for (operation in listOf("gemv", "symv", "trsv", "trmv")) {
                    val before = output.copyOf()
                    val expected = if (operation.startsWith("tr")) x.toDoubleArray() else y.toDoubleArray()
                    when (operation) {
                        "gemv" -> {
                            ReferenceBlas.gemv(0.875, a, x.toDoubleArray(), -0.25, expected, transpose)
                            blas.kernels.gemv(
                                0.875,
                                a,
                                x.values,
                                x.offset,
                                x.stride,
                                -0.25,
                                y.values,
                                y.offset,
                                y.stride,
                                transpose,
                            )
                        }

                        "symv" -> {
                            ReferenceBlas.symv(0.875, a, x.toDoubleArray(), -0.25, expected, transpose)
                            blas.kernels.symv(
                                0.875,
                                a,
                                x.values,
                                x.offset,
                                x.stride,
                                -0.25,
                                y.values,
                                y.offset,
                                y.stride,
                                transpose,
                            )
                        }

                        "trsv" -> {
                            for (i in 0 until 5) y[i] = x[i]
                            ReferenceBlas.trsv(a, expected, true, transpose, true)
                            blas.kernels.trsv(a, y.values, y.offset, y.stride, true, transpose, true)
                        }

                        else -> {
                            for (i in 0 until 5) y[i] = x[i]
                            ReferenceBlas.trmv(a, expected, true, transpose, true)
                            blas.kernels.trmv(a, y.values, y.offset, y.stride, true, transpose, true)
                        }
                    }
                    for (i in 0 until 5) before[y.offset + i * y.stride] = expected[i]
                    assertClose(before, output, "$operation stride $stride")
                }
            }
        }
    }

    @Test
    fun `iamax over an empty vector reports no index`() = withRegionEngines { blas ->
        val empty = DenseVector.zero(0)

        val index = blas.kernels.iamax(empty.values, empty.offset, empty.size, empty.stride)

        assertEquals(-1, index)
    }

    @Test
    fun `matrix vector calls over no columns scale the output`() = withRegionEngines { blas ->
        val a = paddedRegion(4, 0)
        val actual = doubleArrayOf(1.0, -2.0, 3.0, -4.0)
        val expected = actual.copyOf()
        ReferenceBlas.gemv(1.0, contiguousCopy(a), doubleArrayOf(), 0.5, expected)

        blas.kernels.gemv(1.0, a, doubleArrayOf(), 0, 1, 0.5, actual, 0, 1)

        assertClose(expected, actual, "empty inner dimension gemv")
    }

    @Test
    fun `rank updates preserve the unselected triangle and parent entries`() = withRegionEngines { blas ->
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val a = paddedRegion(if (transpose) 4 else 7, if (transpose) 7 else 4)
                val c = paddedRegion(7, 7)
                val expected = contiguousCopy(c)
                val before = c.values.copyOf()
                ReferenceBlas.syrk(-1.0, contiguousCopy(a), transpose, 0.5, expected, lower)

                blas.syrk(-1.0, a, transpose, 0.5, c, lower, scratch = Scratch())

                assertRegionAgreesWithReference(expected, c, before)
            }
        }
    }

    @Test
    fun `rank updates over empty inputs scale the selected triangle`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            val a = paddedRegion(4, 0)
            val c = paddedRegion(4, 4)
            val expected = contiguousCopy(c)
            val before = c.values.copyOf()
            ReferenceBlas.syrk(1.0, contiguousCopy(a), false, 0.5, expected, lower)

            blas.syrk(1.0, a, false, 0.5, c, lower)

            assertRegionAgreesWithReference(expected, c, before)
        }
    }

    @Test
    fun `addressed rank updates preserve padding and unselected entries`() = withRegionEngines { blas ->
        for (stride in intArrayOf(2, -2)) {
            for (lower in booleanArrayOf(false, true)) {
                val x = DenseVector.wrap(DoubleArray(15) { it / 7.0 }, if (stride > 0) 2 else 10, 5, stride)
                val y = DenseVector.wrap(DoubleArray(17) { (it - 3) / 9.0 }, if (stride > 0) 3 else 11, 5, stride)
                for (operation in listOf("ger", "syr", "syr2")) {
                    val a = paddedRegion(5, 5)
                    val before = a.values.copyOf()
                    val expected = contiguousCopy(a)
                    when (operation) {
                        "ger" -> {
                            ReferenceBlas.ger(0.875, x.toDoubleArray(), y.toDoubleArray(), expected)
                            blas.kernels.ger(0.875, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a)
                        }

                        "syr" -> {
                            ReferenceBlas.syr(0.875, x, expected, lower)
                            blas.kernels.syr(0.875, x.values, x.offset, x.stride, a, lower)
                        }

                        else -> {
                            ReferenceBlas.syr2(0.875, x, y, expected, lower)
                            blas.kernels.syr2(
                                0.875, x.values, x.offset, x.stride, y.values, y.offset, y.stride, a, lower,
                            )
                        }
                    }
                    assertRegionAgreesWithReference(expected, a, before, "$operation stride=$stride lower=$lower")
                }
            }
        }
    }

    @Test
    fun `triangular matrix solves agree across flags and diagonal blocks`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    for (unit in booleanArrayOf(false, true)) {
                        val a = paddedRegion(67, 67)
                        val b = paddedRegion(if (right) 5 else 67, if (right) 67 else 5, 2)
                        val expected = contiguousCopy(b)
                        val before = b.values.copyOf()
                        ReferenceBlas.trsm(contiguousCopy(a), expected, lower, transpose, unit, right, 0.875)

                        blas.trsm(a, b, lower, transpose, unit, right, 0.875, scratch = Scratch())

                        assertRegionAgreesWithReference(expected, b, before)
                    }
                }
            }
        }
    }

    @Test
    fun `triangular vector solves agree across flags`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    val a = paddedRegion(7, 7)
                    val actual = DoubleArray(7) { it / 8.0 }
                    val expected = actual.copyOf()
                    ReferenceBlas.trsv(contiguousCopy(a), expected, lower, transpose, unit)

                    blas.kernels.trsv(a, actual, 0, 1, lower, transpose, unit, scratch = Scratch())

                    assertClose(expected, actual, "view vector")
                }
            }
        }
    }

    @Test
    fun `unselected triangles and implicit unit diagonals are never read`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            val a = paddedRegion(6, 6)
            for (j in 0 until 6) for (i in 0 until 6) if (i == j || (i > j) != lower) a[i, j] = Double.NaN
            val b = paddedRegion(6, 3, 2)
            val x = DoubleArray(6) { it / 8.0 }
            val panel = paddedRegion(6, 3, 3)
            val c = paddedRegion(6, 6, 4)
            for (j in 0 until 6) for (i in 0 until 6) if (i != j && (i > j) != lower) c[i, j] = Double.NaN
            val expectedB = contiguousCopy(b)
            val expectedX = x.copyOf()
            val expectedC = contiguousCopy(c)
            val beforeB = b.values.copyOf()
            val beforeC = c.values.copyOf()
            ReferenceBlas.trsm(contiguousCopy(a), expectedB, lower, unitDiag = true)
            ReferenceBlas.trsv(contiguousCopy(a), expectedX, lower, unitDiag = true)
            ReferenceBlas.syrk(-1.0, contiguousCopy(panel), false, 1.0, expectedC, lower)

            blas.trsm(a, b, lower, unitDiag = true, scratch = Scratch())
            blas.kernels.trsv(a, x, 0, 1, lower, unitDiag = true, scratch = Scratch())
            blas.syrk(-1.0, panel, false, 1.0, c, lower, scratch = Scratch())

            assertRegionAgreesWithReference(expectedB, b, beforeB, "unit diagonal trsm")
            assertClose(expectedX, x, "unit diagonal trsv")
            assertRegionAgreesWithReference(expectedC, c, beforeC, "syrk")
        }
    }

    @Test
    fun `structured products address every padded operand and preserve the parent`() = withRegionEngines { blas ->
        for (lower in booleanArrayOf(false, true)) {
            val a = paddedRegion(5, 5)
            val b = paddedRegion(5, 5, 2)
            for (transpose in booleanArrayOf(false, true)) {
                assertStructuredProduct("gemmt lower=$lower t=$transpose", { expected ->
                    ReferenceBlas.gemmt(0.875, a, transpose, b, !transpose, -0.25, expected, lower)
                }) { c -> blas.gemmt(0.875, a, transpose, b, !transpose, -0.25, c, lower) }
                assertStructuredProduct("syr2k lower=$lower t=$transpose", { expected ->
                    ReferenceBlas.syr2k(0.875, a, b, transpose, -0.25, expected, lower)
                }) { c -> blas.syr2k(0.875, a, b, transpose, -0.25, c, lower) }
                for (right in booleanArrayOf(false, true)) {
                    assertStructuredProduct("trmm lower=$lower t=$transpose right=$right", { expected ->
                        ReferenceBlas.trmm(a, expected, lower, transpose, true, right, 0.875)
                    }) { c -> blas.trmm(a, c, lower, transpose, true, right, 0.875) }
                }
            }
            for (right in booleanArrayOf(false, true)) {
                assertStructuredProduct("symm lower=$lower right=$right", { expected ->
                    ReferenceBlas.symm(0.875, a, b, -0.25, expected, lower, right)
                }) { c -> blas.symm(0.875, a, b, -0.25, c, lower, right) }
            }
        }
    }

    private fun assertStructuredProduct(
        context: String,
        reference: (DenseMatrix) -> Unit,
        call: (DenseMatrix) -> Unit,
    ) {
        val c = paddedRegion(5, 5, 3)
        val before = c.values.copyOf()
        val expected = contiguousCopy(c)
        reference(expected)

        call(c)

        assertRegionAgreesWithReference(expected, c, before, context)
    }

    @Test
    fun `zero multipliers overwrite only the destination rectangle`() = withRegionEngines { blas ->
        val a = paddedRegion(4, 4).also { it.values.fill(Double.NaN) }
        val b = paddedRegion(4, 4).also { it.values.fill(Double.NaN) }
        val c = paddedRegion(4, 4).also { it.values.fill(Double.NaN) }
        val before = c.values.copyOf()
        val expected = DenseMatrix.zero(4)
        val workspace = Scratch()

        blas.gemm(0.0, a, false, b, false, 0.0, c, scratch = workspace)

        assertRegionAgreesWithReference(expected, c, before)
        assertEquals(0, workspace.available(16))
        blas.trsm(a, c, lower = true, alpha = 0.0, scratch = workspace)
        assertRegionAgreesWithReference(expected, c, before)
    }

    @Test
    fun `zero rank and vector multipliers skip poisoned inputs`() = withRegionEngines { blas ->
        val a = paddedRegion(4, 4).also { it.values.fill(Double.NaN) }
        val c = paddedRegion(4, 4)
        val before = c.values.copyOf()
        val expected = contiguousCopy(c)
        for (j in 0 until 4) for (i in j until 4) expected[i, j] = 0.0
        val y = DoubleArray(4) { Double.NaN }

        blas.syrk(0.0, a, false, 0.0, c)
        blas.kernels.gemv(0.0, a, DoubleArray(4) { Double.NaN }, 0, 1, 0.0, y, 0, 1)

        assertRegionAgreesWithReference(expected, c, before)
        assertContentEquals(DoubleArray(4), y)
    }

    @Test
    fun `empty shared dimensions scale rectangular destinations`() = withRegionEngines { blas ->
        val a = paddedRegion(4, 0)
        val b = paddedRegion(0, 3)
        val c = paddedRegion(4, 3)
        val expected = contiguousCopy(c)
        val before = c.values.copyOf()
        ReferenceBlas.gemm(1.0, contiguousCopy(a), false, contiguousCopy(b), false, 0.5, expected)

        blas.gemm(1.0, a, false, b, false, 0.5, c)

        assertRegionAgreesWithReference(expected, c, before)
    }

    /** A call whose contract stops before the arithmetic takes no loan, because it stages nothing. */
    @Test
    fun `a call with nothing to compute never reaches the workspace`() {
        val rng = Random(20260931)
        val n = 4
        val workspace = Scratch()
        val a = randomMatrix(n, n, rng)
        val c = randomMatrix(n, n, rng)

        KoblasEngine.scalar.gemm(0.0, a, false, a, false, -0.25, c, scratch = workspace)
        KoblasEngine.scalar.kernels.gemv(
            0.0,
            a,
            DoubleArray(n),
            0,
            1,
            -0.25,
            DoubleArray(n),
            0,
            1,
            false,
            scratch = workspace,
        )

        assertEquals(0, workspace.idleLengths(), "a no-work call borrowed scratch it had no use for")
    }

    // Every operand shares one empty array, so a call testing for an alias before testing for an empty
    // destination would take a staging loan for a result that does not exist.
    @Test
    fun `a structured call with no output borrows nothing`() = withEngines { blas ->
        val workspace = Scratch()
        val shared = DoubleArray(0)
        val tall = DenseMatrix.wrap(0, 4, shared)
        val wide = DenseMatrix.wrap(4, 0, shared)
        val square = DenseMatrix.wrap(0, 0, shared)
        val destination = DenseMatrix.wrap(0, 4, shared)

        blas.gemmt(0.875, tall, false, wide, false, -0.25, square, scratch = workspace)
        blas.syrk(0.875, tall, false, -0.25, square, scratch = workspace)
        blas.syr2k(0.875, tall, tall, false, -0.25, square, scratch = workspace)
        blas.symm(0.875, square, tall, -0.25, destination, scratch = workspace)
        blas.trsm(square, tall, lower = true, scratch = workspace)
        blas.trmm(square, tall, lower = true, scratch = workspace)

        assertEquals(0, workspace.idleLengths(), "a call with no output borrowed scratch it had no use for")
    }

    @Test
    fun `product aliases agree with the reference and share one snapshot`() = withEngines { blas ->
        assertDenseProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `rectangular product aliases retain their operand shapes`() = withEngines { blas ->
        assertRectangularProductAliasesAgreeWithReference(blas)
    }

    @Test
    fun `empty calls validate shapes and leave backing buffers untouched`() = withRegionEngines { blas ->
        val a = paddedRegion(0, 3)
        val b = paddedRegion(3, 4)
        val c = paddedRegion(0, 4)
        val before = c.values.copyOf()

        blas.gemm(1.0, a, false, b, false, 0.0, c)

        assertContentEquals(before, c.values)
        assertFailsWith<DimensionMismatch> { blas.gemm(0.0, a, false, b, false, 0.0, paddedRegion(1, 4)) }
    }

    @Test
    fun `gemv and gemm reject incompatible shapes`() {
        assertFailsWith<DimensionMismatch> {
            koblas.gemm(1.0, DenseMatrix(2, 3), false, DenseMatrix(2, 2), false, 0.0, DenseMatrix.zero(2, 2))
        }
        assertFailsWith<DimensionMismatch> {
            koblas.kernels.gemv(
                1.0,
                DenseMatrix(2, 3),
                DoubleArray(2),
                0,
                1,
                0.0,
                DoubleArray(2),
                0,
                1,
            )
        }
    }

    @Test
    fun `no work products validate shapes before returning`() {
        for ((alpha, depth) in listOf(0.0 to 3, 0.875 to 0)) {
            val a = DenseMatrix(4, depth)
            val b = DenseMatrix(depth, 4)
            val c = DenseMatrix(4, 4, DoubleArray(16) { 7.0 })
            val wrongOutput = DenseMatrix(4, 3, DoubleArray(12) { 7.0 })
            val calls = listOf<() -> Unit>(
                { koblas.gemm(alpha, a, false, DenseMatrix(depth + 1, 4), false, 1.0, c) },
                { koblas.gemm(alpha, a, false, b, false, 1.0, wrongOutput) },
                { koblas.gemmt(alpha, a, false, DenseMatrix(depth + 1, 4), false, 1.0, c) },
                { koblas.gemmt(alpha, a, false, b, false, 1.0, wrongOutput) },
                { koblas.syrk(alpha, DenseMatrix(3, depth), false, 1.0, c) },
                { koblas.syrk(alpha, a, false, 1.0, wrongOutput) },
                { koblas.syr2k(alpha, a, DenseMatrix(4, depth + 1), false, 1.0, c) },
                { koblas.syr2k(alpha, a, a, false, 1.0, wrongOutput) },
            )

            for (call in calls) assertFailsWith<DimensionMismatch> { call() }

            assertContentEquals(DoubleArray(16) { 7.0 }, c.values)
            assertContentEquals(DoubleArray(12) { 7.0 }, wrongOutput.values)
        }
    }

    /**
     * A square operand is the whole of what a symmetric routine can mean, so the rejection is the contract.
     * The shape is checked before anything reaches a library.
     */
    @Test
    fun `symv refuses a non square matrix`() {
        val n = 17
        assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
            koblas.kernels.symv(1.0, DenseMatrix.zero(n, n + 1), DoubleArray(n), 0, 1, 0.0, DoubleArray(n), 0, 1)
        }
        assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
            koblas.kernels.symv(
                1.0, DenseMatrix.zero(n + 1, n), DoubleArray(n + 1), 0, 1, 0.0, DoubleArray(n + 1), 0, 1,
            )
        }
    }

    // Scaling C is part of the operation, so a call that cannot go through must not have done it. C matches
    // B, so the shape that fails is A against B: too few columns from the right, too few rows from the left.
    @Test
    fun `symm leaves C alone when the operands do not line up`() {
        for (right in booleanArrayOf(true, false)) {
            for (beta in doubleArrayOf(0.0, 2.0)) {
                val a = DenseMatrix.diagonal(if (right) 2 else 3)
                val b = DenseMatrix(2, 3)
                val c = DenseMatrix(2, 3)
                c.values.fill(7.0)

                assertFailsWith<DimensionMismatch>("symm right=$right beta=$beta should reject these shapes") {
                    koblas.symm(1.0, a, b, beta, c, lower = true, right = right)
                }

                assertTrue(
                    c.values.all { it == 7.0 },
                    "symm right=$right beta=$beta scaled C before rejecting the call: ${c.values.toList()}",
                )
            }
        }
    }

    @Test
    fun `trsv and trsm validate shapes`() {
        assertFailsWith<DimensionMismatch> {
            koblas.kernels.trsv(DenseMatrix(2, 3), DoubleArray(2), 0, 1, lower = true)
        }
        assertFailsWith<DimensionMismatch> {
            koblas.kernels.trsv(DenseMatrix(3, 3), DoubleArray(2), 0, 1, lower = true)
        }
        assertFailsWith<DimensionMismatch> { koblas.trsm(DenseMatrix(3, 3), DenseMatrix(2, 4), lower = true) }
    }

    @Test
    fun `GER agrees with the reference on shrinking supports and extreme finite inputs`() = withKoblasEngine { engine ->
        assertShrinkingGerAgreesWithReference { x, y, a ->
            engine.kernels.ger(
                -1.0,
                x,
                0,
                1,
                y,
                0,
                1,
                a,
            )
        }
        assertShrinkingGerAgreesWithReference { x, y, a ->
            val start = x.indexOfFirst { it != 0.0 }
            val extent = a.rows - start
            engine.kernels.ger(-1.0, x, y, a, start, start, extent, extent, start, start)
        }
    }

    @Test
    fun `windowed GER agrees with the reference with independent offsets and parent strides`() =
        withKoblasEngine { engine ->
            for ((rows, columns) in listOf(1 to 1, 3 to 2, 5 to 4)) {
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                val x = DoubleArray(10) { (it - 4) / 16.0 }
                val y = DoubleArray(11) { (5 - it) / 8.0 }
                val originalX = x.copyOf()
                val originalY = y.copyOf()
                val expected = referenceWindow(a, x, y, rows, columns)

                engine.kernels.ger(-0.875, x, y, a, 2, 3, rows, columns, 1, 4)

                assertWindowAgreesWithReference(expected, a, rows, columns)
                assertContentEquals(originalX, x)
                assertContentEquals(originalY, y)
            }
        }

    @Test
    fun `windowed GER snapshots either or both aliased vector ranges`() = withKoblasEngine { engine ->
        for (alias in 1..3) {
            val workspace = Scratch()
            repeat(2) { _ ->
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                val x = if (alias != 2) a.values else DoubleArray(10) { (it - 4) / 16.0 }
                val y = if (alias != 1) a.values else DoubleArray(11) { (5 - it) / 8.0 }
                val xOffset = if (alias != 2) 29 else 1
                val yOffset = if (alias != 1) 30 else 4
                val expected = referenceWindow(a, x, y, 3, 2, xOffset, yOffset)

                engine.kernels.ger(-0.875, x, y, a, 2, 3, 3, 2, xOffset, yOffset, scratch = workspace)

                assertWindowAgreesWithReference(expected, a, 3, 2)
                assertEquals(if (alias != 2) 1 else 0, workspace.available(3))
                assertEquals(if (alias != 1) 1 else 0, workspace.available(2))
                assertEquals(0, workspace.available(a.values.size))
            }
        }
    }

    @Test
    fun `windowed GER validates every range before writing even with zero alpha`() = withKoblasEngine { engine ->
        val valid = intArrayOf(2, 3, 3, 2, 1, 4)
        for (alpha in doubleArrayOf(0.0, -1.0)) {
            for (index in valid.indices) {
                for (invalid in intArrayOf(-1, Int.MAX_VALUE)) {
                    val args = valid.copyOf().also { it[index] = invalid }
                    val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { it.toDouble() })
                    val before = a.values.copyOf()

                    assertFailsWith<DimensionMismatch> {
                        engine.kernels.ger(
                            alpha, DoubleArray(10), DoubleArray(11), a,
                            args[0], args[1], args[2], args[3], args[4], args[5],
                        )
                    }

                    assertContentEquals(before, a.values)
                }
            }
        }
    }

    @Test
    fun `empty windows and zero alpha do not read entries or borrow scratch`() = withKoblasEngine { engine ->
        for (shape in listOf(0 to 0, 0 to 2, 3 to 0, 3 to 2)) {
            val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { Double.NaN })
            val workspace = Scratch()
            val alpha = if (shape.first > 0 && shape.second > 0) 0.0 else Double.NaN

            engine.kernels.ger(
                alpha, a.values, a.values, a,
                9 - shape.first, 8 - shape.second, shape.first, shape.second,
                72 - shape.first, 72 - shape.second,
                scratch = workspace,
            )

            assertContentEquals(DoubleArray(72) { Double.NaN }, a.values)
            assertEquals(0, workspace.idleLengths())
        }
    }

    /** The scalar and vector compositions and the platform default, each once. */
    private fun withKoblasEngine(body: (KoblasEngine) -> Unit) {
        listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd, koblas).distinct().forEach(body)
    }

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
            DoubleArray(rows * columns) { index -> a[2 + index % rows, 3 + index / rows] },
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

    private companion object {
        /** Tall, wide, single row, single column and single entry results, with and without a unit depth. */
        val GEMM_SHAPES = listOf(
            Triple(5, 7, 4),
            Triple(3, 7, 6),
            Triple(7, 5, 6),
            Triple(1, 7, 6),
            Triple(1, 1, 9),
            Triple(6, 7, 1),
            Triple(9, 1, 1),
            Triple(1, 7, 1),
        )
    }

    private fun dense(vararg values: Double) = DenseVector.of(values)

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
    fun `shared buffer axpy agrees with the scalar snapshot reference`() {
        for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd, koblas)) {
            for ((sourceOffset, destinationOffset, stride) in listOf(
                Triple(0, 1, 2),
                Triple(8, 9, -2),
                Triple(0, 5, 1),
                Triple(0, 1, 1),
            )) {
                val backing = DoubleArray(10) { it + 0.25 }
                val source = DenseVector.wrap(backing, sourceOffset, 5, stride)
                val destination = DenseVector.wrap(backing, destinationOffset, 5, stride)
                val expected = backing.copyOf()
                KoblasEngine.scalar.kernels.axpy(
                    expected,
                    destinationOffset,
                    0.75,
                    source.toDoubleArray(),
                    0,
                    5,
                    stride,
                )

                engine.axpyVector(0.75, source, destination)

                assertAxpyAgreesWithReference(expected, backing)
            }
        }
    }

    @Test
    fun `shared buffer copy preserves the snapshot sequence and padding`() {
        for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd, koblas)) {
            for ((sourceOffset, destinationOffset, stride) in listOf(
                Triple(0, 1, 2),
                Triple(8, 9, -2),
                Triple(0, 5, 1),
                Triple(0, 1, 1),
                Triple(1, 0, 1),
                Triple(2, 7, 1),
                Triple(9, 8, -1),
            )) {
                val backing = DoubleArray(14) { it + 0.25 }
                val source = DenseVector.wrap(backing, sourceOffset, 5, stride)
                val destination = DenseVector.wrap(backing, destinationOffset, 5, stride)
                val expected = backing.copyOf()
                val snapshot = source.toDoubleArray()
                for (i in snapshot.indices) expected[destinationOffset + i * stride] = snapshot[i]

                engine.copyVector(source, destination)

                assertContentEquals(expected, backing)
            }
        }
    }

    @Test
    fun `matvec calls snapshot vector aliases before writing`() {
        val matrix = DenseMatrix.diagonal(3)
        for (reversed in booleanArrayOf(false, true)) {
            for (symmetric in booleanArrayOf(false, true)) {
                val destination = doubleArrayOf(1.0, 2.0, 3.0)
                val original = destination.copyOf()
                // A call that scaled the destination before reading its input would see the scaled values.
                val source =
                    if (reversed) DenseVector.wrap(destination, 2, 3, -1) else DenseVector.wrap(destination, 0, 3, 1)
                val expected = DoubleArray(3) { source[it] + 0.5 * original[it] }

                if (symmetric) {
                    koblas.kernels.symv(
                        1.0,
                        matrix,
                        source.values,
                        source.offset,
                        source.stride,
                        0.5,
                        (DenseVector.wrap(destination)).values,
                        (DenseVector.wrap(destination)).offset,
                        (DenseVector.wrap(destination)).stride,
                    )
                } else {
                    koblas.kernels.gemv(
                        1.0,
                        matrix,
                        source.values,
                        source.offset,
                        source.stride,
                        0.5,
                        (DenseVector.wrap(destination)).values,
                        (DenseVector.wrap(destination)).offset,
                        (DenseVector.wrap(destination)).stride,
                    )
                }

                assertContentEquals(expected, destination)
            }
        }
    }

    @Test
    fun `gemv snapshots matrix aliases before writing`() {
        val destination = doubleArrayOf(1.0, 2.0, 3.0)
        val matrix = DenseMatrix.wrap(3, 1, destination)

        koblas.kernels.gemv(
            1.0,
            matrix,
            (dense(2.0)).values,
            (dense(2.0)).offset,
            (dense(2.0)).stride,
            0.5,
            (DenseVector.wrap(destination)).values,
            (DenseVector.wrap(destination)).offset,
            (DenseVector.wrap(destination)).stride,
        )

        assertContentEquals(doubleArrayOf(2.5, 5.0, 7.5), destination)
    }

    @Test
    fun `symv snapshots its matrix alias before writing`() {
        val destination = doubleArrayOf(2.0)
        val matrix = DenseMatrix.wrap(1, 1, destination)

        koblas.kernels.symv(
            1.0,
            matrix,
            (dense(3.0)).values,
            (dense(3.0)).offset,
            (dense(3.0)).stride,
            0.5,
            (DenseVector.wrap(destination)).values,
            (DenseVector.wrap(destination)).offset,
            (DenseVector.wrap(destination)).stride,
        )

        assertContentEquals(doubleArrayOf(7.0), destination)
    }

    @Test
    fun `gemv honours beta on a matrix with no columns`() {
        val matrix = DenseMatrix.zero(3, 0)
        val destination = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)

        koblas.kernels.gemv(
            1.0,
            matrix,
            (DenseVector.zero(0)).values,
            (DenseVector.zero(0)).offset,
            (DenseVector.zero(0)).stride,
            0.0,
            (DenseVector.wrap(destination)).values,
            (DenseVector.wrap(destination)).offset,
            (DenseVector.wrap(destination)).stride,
        )

        assertContentEquals(doubleArrayOf(0.0, 0.0, 0.0), destination)
    }

    /** Stepped operands reach addressed panels without borrowing a gather buffer. */
    @Test
    fun `a strided matvec operand needs no gather buffer`() {
        val n = 6
        val backing = DoubleArray(2 * n) { 1.0 + it }
        val x = DenseVector.wrap(backing, 0, n, 2)
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 5) * 0.25 })
        val expected = DoubleArray(n).also {
            koblas.kernels.gemv(
                1.0,
                a,
                x.values,
                x.offset,
                x.stride,
                0.0,
                (DenseVector.wrap(it)).values,
                (DenseVector.wrap(it)).offset,
                (DenseVector.wrap(it)).stride,
            )
        }
        val workspace = Scratch()
        val destination = DoubleArray(n)

        koblas.kernels.gemv(
            1.0,
            a,
            x.values,
            x.offset,
            x.stride,
            0.0,
            (DenseVector.wrap(destination)).values,
            (DenseVector.wrap(destination)).offset,
            (DenseVector.wrap(destination)).stride,
            scratch = workspace,
        )

        assertContentEquals(expected, destination)
        assertEquals(0, workspace.available(n), "the addressed call borrowed a gather")
    }
}
