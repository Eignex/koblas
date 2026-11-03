package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.workspace.*
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The CBLAS layer against the scalar oracle: direct Level 1 calls on one installed vendor, and composed
 * Level 2 and 3 calls on every installed vendor. A host without a library reports the coverage it skipped.
 *
 * Real bindings expose operand, stride and flag translation, which is what these check. A loose tolerance
 * admits the library's accumulation order while catching incorrect translation, and the composed fixtures
 * admit every size so translation is isolated from the platform cutoffs, which StandardEngineProfilesTest covers.
 */
class CblasTest {
    private val scalar = BuiltinEngines.scalar

    @Test
    fun `singleton norm accepts extreme strides`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val vector = StridedVector(values, 1, 1, stride)
            val expected = scalar.nrm2(values, 1, 1, stride)

            val actual = vendor.nrm2(vector)

            assertNormAgreesWithReference(expected, actual, "${vendor.vendor}, stride=$stride")
        }
    }

    @Test
    fun `singleton absolute sum accepts extreme strides`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val vector = StridedVector(values, 1, 1, stride)
            val expected = scalar.asum(values, 1, 1, stride)

            val actual = vendor.asum(vector)

            assertAsumAgreesWithReference(expected, actual, "${vendor.vendor}, stride=$stride")
        }
    }

    @Test
    fun `singleton scaling accepts extreme strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val expected = values.copyOf()
            val vector = StridedVector(values, 1, 1, stride)
            scalar.scale(expected, 1, -0.75, 1, stride)

            vendor.scale(-0.75, vector)

            assertScalAgreesWithReference(expected, values, "${vendor.vendor}, stride=$stride")
        }
    }

    @Test
    fun `dot agrees with the scalar oracle under independent strides`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expected = scalar.dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)

            val actual = vendor.dot(x, y)

            assertAgreesWithReference(expected, actual, context(vendor, "dot", x, y))
        }
    }

    @Test
    fun `norm agrees with the scalar oracle under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.nrm2(x.values, x.offset, x.size, x.stride)

            val actual = vendor.nrm2(x)

            assertAgreesWithReference(expected, actual, context(vendor, "nrm2", x))
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.asum(x.values, x.offset, x.size, x.stride)

            val actual = vendor.asum(x)

            assertAgreesWithReference(expected, actual, context(vendor, "asum", x))
        }
    }

    @Test
    fun `maximum index selects a largest magnitude under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.iamax(x.values, x.offset, x.size, x.stride)

            val actual = vendor.iamax(x)

            assertIamaxHonoursItsContract(expected, actual, x, context(vendor, "iamax", x))
        }
    }

    @Test
    fun `a backward vector reports the last of several equal maxima`() {
        val vendor = vendor ?: return skipped()
        for (stride in intArrayOf(1, 2)) {
            val forward = strideVector(stride, 3, 24)
            val entries = doubleArrayOf(1.0, 5.0, 2.0, -5.0, 3.0, 0.5)
            for (i in entries.indices) forward.values[forward.offset + i * stride] = entries[i]
            val backward = StridedVector(
                forward.values,
                forward.offset + (forward.size - 1) * stride,
                forward.size,
                -stride,
            )
            val expected = scalar.iamax(forward.values, forward.offset, forward.size, stride)

            val actualForward = vendor.iamax(forward)
            val actualBackward = vendor.iamax(backward)

            assertEquals(expected, actualForward, context(vendor, "iamax forward tie", forward))
            // The binding walks backward vectors in physical ascending order, then remaps the index.
            assertEquals(backward.size - 1 - expected, actualBackward, context(vendor, "iamax backward tie", backward))
        }
    }

    @Test
    fun `scaling agrees with the scalar oracle without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = x.values.copyOf()
            scalar.scale(expected, x.offset, -0.75, x.size, x.stride)

            vendor.scale(-0.75, x)

            assertAgreesWithReference(expected, x.values, context(vendor, "scal", x))
        }
    }

    @Test
    fun `axpy agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val originalX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            scalar.axpy(expectedY, y.offset, 0.5, x.values, x.offset, x.size, y.stride, x.stride)

            vendor.axpy(0.5, x, y)

            assertContentEquals(originalX, x.values, context(vendor, "axpy source", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "axpy destination", x, y))
        }
    }

    @Test
    fun `copy preserves logical order under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val originalX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            for (i in 0 until x.size) expectedY[y.offset + i * y.stride] = x[i]

            vendor.copy(x, y)

            assertContentEquals(originalX, x.values, context(vendor, "copy source", x, y))
            assertContentEquals(expectedY, y.values, context(vendor, "copy destination", x, y))
        }
    }

    @Test
    fun `swap agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expectedX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            scalar.swap(expectedX, x.offset, expectedY, y.offset, x.size, x.stride, y.stride)

            vendor.swap(x, y)

            assertAgreesWithReference(expectedX, x.values, context(vendor, "swap first", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "swap second", x, y))
        }
    }

    @Test
    fun `rotation agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expectedX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            val packedX = x.toDoubleArray()
            val packedY = y.toDoubleArray()
            scalar.rot(packedX, 0, packedY, 0, x.size, 0.6, 0.8)
            for (i in 0 until x.size) {
                expectedX[x.offset + i * x.stride] = packedX[i]
                expectedY[y.offset + i * y.stride] = packedY[i]
            }

            vendor.rot(x, y, 0.6, 0.8)

            assertAgreesWithReference(expectedX, x.values, context(vendor, "rot first", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "rot second", x, y))
        }
    }

    // Composed Level 2 and 3 calls, through each installed vendor admitted at every size.

    @Test
    fun `host GER agrees with the reference on shrinking supports and extreme finite inputs`() =
        withHost("GER shrinking supports") { blas ->
            assertShrinkingGerAgreesWithReference { x, y, a -> blas.ger(-1.0, x, y, a) }
        }

    @Test
    fun `a symmetric product from the right is the other side and not the same call`() =
        withHost("the right-side symmetric product") { blas ->
            val rng = Random(14)
            val (full, poisoned) = poisonedSymmetric(rng, ORDER, lower = true)
            val b = randomMatrix(SIDES, ORDER, rng)
            val expected = randomMatrix(SIDES, ORDER, Random(15))
            val actual = DenseMatrix.wrap(SIDES, ORDER, expected.values.copyOf())
            ReferenceBlas.symm(ALPHA, full, b, BETA, expected, lower = true, right = true)

            blas.symm(ALPHA, poisoned, b, BETA, actual, lower = true, right = true)

            assertClose(expected, actual, "symm right", TOLERANCE)
        }

    @Test
    fun `the rank updates write only the selected triangle`() = withHost("the rank updates") { blas ->
        val rng = Random(16)
        for (lower in booleanArrayOf(true, false)) {
            val x = DenseVector.wrap(randomVector(ORDER, rng))
            val y = DenseVector.wrap(randomVector(ORDER, rng))
            val base = randomMatrix(ORDER, ORDER, Random(17))
            val expectedSyr = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            val expectedSyr2 = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            ReferenceBlas.syr(ALPHA, x, expectedSyr, lower)
            ReferenceBlas.syr2(ALPHA, x, y, expectedSyr2, lower)

            val actualSyr = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            val actualSyr2 = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            blas.syr(ALPHA, x, actualSyr, lower)
            blas.syr2(ALPHA, x, y, actualSyr2, lower)

            assertClose(expectedSyr, actualSyr, "syr lower=$lower", TOLERANCE)
            assertClose(expectedSyr2, actualSyr2, "syr2 lower=$lower", TOLERANCE)
        }
    }

    @Test
    fun `a general rank update covers the whole destination`() = withHost("ger") { blas ->
        val rng = Random(18)
        val x = randomVector(ROWS, rng)
        val y = randomVector(COLUMNS, rng)
        val base = randomMatrix(ROWS, COLUMNS, rng)
        val expected = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
        ReferenceBlas.ger(ALPHA, x, y, expected)

        val actual = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
        blas.ger(ALPHA, x, y, actual)

        assertClose(expected, actual, "ger", TOLERANCE)
    }

    // A NaN on an implied diagonal makes the unit-diagonal flag observable: a library reading the stored
    // diagonal comes back all NaN rather than slightly wrong.
    @Test
    fun `the triangular vector routines cover all eight traversals`() =
        withHost("the triangular vector routines") { blas ->
            val rng = Random(19)
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (poisoned, explicit) = poisonedTriangle(rng, ORDER, lower, unitDiag)
                        val b = randomVector(ORDER, rng)
                        val where = "lower=$lower transpose=$transpose unit=$unitDiag"

                        val expectedSolve = b.copyOf()
                        ReferenceBlas.trsv(explicit, expectedSolve, lower, transpose, unitDiag)
                        val actualSolve = b.copyOf()
                        blas.trsv(poisoned, actualSolve, lower, transpose, unitDiag)

                        val expectedMultiply = b.copyOf()
                        ReferenceBlas.trmv(explicit, expectedMultiply, lower, transpose, unitDiag)
                        val actualMultiply = b.copyOf()
                        blas.trmv(poisoned, actualMultiply, lower, transpose, unitDiag)

                        assertClose(expectedSolve, actualSolve, "trsv $where", TOLERANCE)
                        assertClose(expectedMultiply, actualMultiply, "trmv $where", TOLERANCE)
                    }
                }
            }
        }

    @Test
    fun `the triangular matrix routines cover both sides and both traversals`() =
        withHost("the triangular matrix routines") { blas ->
            val rng = Random(20)
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        val (poisoned, explicit) = poisonedTriangle(rng, ORDER, lower, unitDiag = false)
                        val rows = if (right) SIDES else ORDER
                        val columns = if (right) ORDER else SIDES
                        val base = randomMatrix(rows, columns, rng)
                        val where = "lower=$lower transpose=$transpose right=$right"

                        val expectedSolve = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        ReferenceBlas.trsm(explicit, expectedSolve, lower, transpose, false, right, ALPHA)
                        val actualSolve = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        blas.trsm(poisoned, actualSolve, lower, transpose, false, right, ALPHA)

                        val expectedMultiply = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        ReferenceBlas.trmm(explicit, expectedMultiply, lower, transpose, false, right, ALPHA)
                        val actualMultiply = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        blas.trmm(poisoned, actualMultiply, lower, transpose, false, right, ALPHA)

                        assertClose(expectedSolve, actualSolve, "trsm $where", TOLERANCE)
                        assertClose(expectedMultiply, actualMultiply, "trmm $where", TOLERANCE)
                    }
                }
            }
        }

    @Test
    fun `a product agrees on every combination of transposes`() = withHost("the product") { blas ->
        val rng = Random(21)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) randomMatrix(DEPTH, ROWS, rng) else randomMatrix(ROWS, DEPTH, rng)
                val b = if (transposeB) randomMatrix(COLUMNS, DEPTH, rng) else randomMatrix(DEPTH, COLUMNS, rng)
                val base = randomMatrix(ROWS, COLUMNS, rng)
                val expected = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
                ReferenceBlas.gemm(ALPHA, a, transposeA, b, transposeB, BETA, expected)

                val actual = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
                blas.gemm(ALPHA, a, transposeA, b, transposeB, BETA, actual)

                assertClose(expected, actual, "gemm transA=$transposeA transB=$transposeB", TOLERANCE)
            }
        }
    }

    @Test
    fun `the triangle selected products write one triangle and leave the other alone`() =
        withHost("the triangle-selected products") { blas ->
            val rng = Random(22)
            for (lower in booleanArrayOf(true, false)) {
                val a = randomMatrix(ORDER, DEPTH, rng)
                val b = randomMatrix(DEPTH, ORDER, rng)
                val base = randomMatrix(ORDER, ORDER, rng)
                val where = "lower=$lower"

                val expectedGemmt = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.gemmt(ALPHA, a, false, b, false, BETA, expectedGemmt, lower)
                val actualGemmt = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.gemmt(ALPHA, a, false, b, false, BETA, actualGemmt, lower)

                val expectedSyrk = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.syrk(ALPHA, a, false, BETA, expectedSyrk, lower)
                val actualSyrk = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.syrk(ALPHA, a, false, BETA, actualSyrk, lower)

                val other = randomMatrix(ORDER, DEPTH, rng)
                val expectedSyr2k = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.syr2k(ALPHA, a, other, false, BETA, expectedSyr2k, lower)
                val actualSyr2k = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.syr2k(ALPHA, a, other, false, BETA, actualSyr2k, lower)

                assertClose(expectedGemmt, actualGemmt, "gemmt $where", TOLERANCE)
                assertClose(expectedSyrk, actualSyrk, "syrk $where", TOLERANCE)
                assertClose(expectedSyr2k, actualSyr2k, "syr2k $where", TOLERANCE)
            }
        }

    // An operand that is the destination's own buffer, which a whole-call binding cannot be handed, so the
    // answer has to be the one the definition gives for two separate matrices.
    @Test
    fun `an operand sharing the destination gives the answer two separate operands would`() =
        withHost("the staged aliases") { blas ->
            val rng = Random(23)
            val workspace = Workspace()
            val square = wellConditioned(ORDER, rng)
            val separate = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())

            val expectedProduct = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            ReferenceBlas.gemm(ALPHA, separate, false, separate, false, BETA, expectedProduct)
            val product = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            blas.withWorkspace(workspace).gemm(ALPHA, product, false, product, false, BETA, product)

            val expectedSolve = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            ReferenceBlas.trsm(separate, expectedSolve, lower = true, alpha = ALPHA)
            val solve = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            blas.withWorkspace(workspace).trsm(solve, solve, lower = true, alpha = ALPHA)

            // The first column is the vector operand and the destination at once.
            val expectedGemv = DoubleArray(ORDER)
            ReferenceBlas.gemv(ALPHA, separate, square.values.copyOf(ORDER), 0.0, expectedGemv)
            val matrix = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            val column = StridedVector(matrix.values, 0, ORDER)
            blas.withWorkspace(workspace).gemv(ALPHA, matrix, column, 0.0, column)

            assertClose(expectedProduct, product, "gemm over one buffer", TOLERANCE)
            assertClose(expectedSolve, solve, "trsm over one buffer", TOLERANCE)
            assertClose(expectedGemv, column.toDoubleArray(), "gemv over one buffer", TOLERANCE)
        }

    /** Each installed vendor behind the portable composition, admitted at every size. */
    private inline fun withHost(what: String, block: (KoblasEngine) -> Unit) {
        val hosts = Vendor.entries.mapNotNull { BuiltinEngines.vendor(it) }
        if (hosts.isEmpty()) {
            println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
            return
        }
        for (host in hosts) block(engineWithHost(engineWithKernels(PortableEngine, PortableEngine), host))
    }

    private fun forEachStride(body: (StridedVector) -> Unit) {
        for (stride in MULTI_ENTRY_STRIDES) body(strideVector(stride, 3, 24))
    }

    private fun forEachStridePair(body: (StridedVector, StridedVector) -> Unit) {
        for (xStride in MULTI_ENTRY_STRIDES) {
            for (yStride in MULTI_ENTRY_STRIDES) {
                body(strideVector(xStride, 3, 24), strideVector(yStride, 7, 25))
            }
        }
    }

    private fun strideVector(stride: Int, padding: Int, seed: Int): StridedVector {
        val size = 6
        val offset = padding + if (stride < 0) (size - 1) * -stride else 0
        val length = maxOf(32, 2 * padding + (size - 1) * abs(stride) + 1)
        return StridedVector(vendorValues(length, seed), offset, size, stride)
    }

    private fun context(vendor: KoblasEngine, operation: String, x: StridedVector, y: StridedVector? = null): String =
        "${vendor.vendor} $operation x offset=${x.offset} stride=${x.stride}" +
            y?.let { " y offset=${it.offset} stride=${it.stride}" }.orEmpty()

    private fun assertAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertClose(expected, actual, context)

    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertClose(expected, actual, context)

    private fun assertIamaxHonoursItsContract(expected: Int, actual: Int, x: StridedVector, context: String) {
        assertTrue(actual in 0 until x.size, "$context returned index $actual")
        assertEquals(abs(x[expected]), abs(x[actual]), context)
    }

    private fun skipped() {
        println("SKIPPED: no CBLAS library installed; direct vendor calls were not verified")
    }

    private fun assertNormAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertEquals(expected, actual, context)

    private fun assertAsumAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertEquals(expected, actual, context)

    private fun assertScalAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertContentEquals(expected, actual, context)

    private companion object {
        // Some vendors special-case singletons and hide a negative increment; prefer the affected binding.
        val vendor = BuiltinEngines.vendor(
            Vendor.OpenBlas,
        ) ?: Vendor.entries.firstNotNullOfOrNull {
            BuiltinEngines.vendor(
                it,
            )
        }

        // Direct KoblasEngine calls bypass the native engine crossover even for these six-entry windows. The
        // widest strides put consecutive entries pages apart in either direction.
        val MULTI_ENTRY_STRIDES = intArrayOf(1, 2, -1, -2, 4099, -4099)
        val STRIDES = intArrayOf(Int.MIN_VALUE, -3, -1, 1, 3, Int.MAX_VALUE)

        const val ROWS = 7
        const val COLUMNS = 5
        const val DEPTH = 6
        const val ORDER = 6
        const val SIDES = 4
        const val ALPHA = 0.75
        const val BETA = -0.5

        /** Loose enough for another accumulation order, far tighter than any mistranslated flag. */
        const val TOLERANCE = 1e-9
    }
}
