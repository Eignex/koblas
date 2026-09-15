package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmVendorBlasTest {
    private fun values(size: Int, seed: Int = 1): DoubleArray {
        var state = seed
        return DoubleArray(size) {
            state = state * 1_103_515_245 + 12_345
            ((state ushr 8) % 1000) / 250.0 - 2.0
        }
    }

    @Test
    fun `gemm over whole matrices agrees with the reference`() = withVendor { blas ->
        val a = MatrixWindow(values(12, 1), 3, 4)
        val b = MatrixWindow(values(20, 2), 4, 5)
        val storage = values(15, 3)
        val c = MatrixWindow(storage, 3, 5)
        val expected = oracleGemm(0.75, a, b, -0.25, c)

        blas.gemm(0.75, a, b, -0.25, c)

        assertAgreesWithReference(expected, c.toPacked(), "gemm")
    }

    @Test
    fun `gemm over offset submatrices keeps the leading dimensions`() = withVendor { blas ->
        val parent = values(400, 4)
        val a = MatrixWindow(parent, 3, 4, offset = 17, rowStride = 1, columnStride = 20)
        val b = MatrixWindow(parent, 4, 5, offset = 140, rowStride = 1, columnStride = 20)
        val c = MatrixWindow(parent, 3, 5, offset = 260, rowStride = 1, columnStride = 20)
        val expected = oracleGemm(1.5, a, b, 0.5, c)
        val untouched = parent.copyOf()

        blas.gemm(1.5, a, b, 0.5, c)

        assertAgreesWithReference(expected, c.toPacked(), "gemm submatrix")
        assertTrue(
            blas.routeOf(VendorOperation.Gemm, listOf(a, b, c)).adapter?.contains("packed copy") != true,
            "a column-major submatrix should not have been packed",
        )
        for (index in parent.indices) {
            val inside = (0 until 5).any { column -> index - 260 - column * 20 in 0 until 3 }
            if (!inside) assertEquals(untouched[index], parent[index], "storage outside the window changed at $index")
        }
    }

    @Test
    fun `a transposed operand reaches the vendor without being copied`() = withVendor { blas ->
        val a = MatrixWindow(values(12, 5), 4, 3).transpose()
        val b = MatrixWindow(values(20, 6), 4, 5)
        val c = MatrixWindow(values(15, 7), 3, 5)
        val expected = oracleGemm(1.0, a, b, 0.0, c)

        blas.gemm(1.0, a, b, 0.0, c)

        assertAgreesWithReference(expected, c.toPacked(), "gemm transposed")
        assertTrue(
            blas.routeOf(VendorOperation.Gemm, listOf(a, b, c)).adapter?.contains("packed copy") != true,
            "a transposed panel reconciles with a flag and should not have been packed",
        )
    }

    @Test
    fun `a window blas cannot address is staged and still correct`() = withVendor { blas ->
        val parent = values(400, 8)
        val a = MatrixWindow(parent, 3, 4, offset = 0, rowStride = 2, columnStride = 40)
        val b = MatrixWindow(values(20, 9), 4, 5)
        val c = MatrixWindow(values(15, 10), 3, 5)
        val expected = oracleGemm(1.0, a, b, 0.0, c)

        blas.gemm(1.0, a, b, 0.0, c)

        assertAgreesWithReference(expected, c.toPacked(), "gemm staged")
        assertTrue(
            blas.routeOf(VendorOperation.Gemm, listOf(a, b, c)).adapter!!.contains("packed copy"),
            "the route should admit the copy it made",
        )
    }

    @Test
    fun `beta zero overwrites the destination without reading it`() = withVendor { blas ->
        val a = MatrixWindow(values(12, 11), 3, 4)
        val b = MatrixWindow(values(20, 12), 4, 5)
        val poisoned = DoubleArray(15) { Double.NaN }
        val c = MatrixWindow(poisoned, 3, 5)
        val clean = MatrixWindow(DoubleArray(15), 3, 5)
        blas.gemm(1.0, a, b, 0.0, clean)

        blas.gemm(1.0, a, b, 0.0, c)

        assertAgreesWithReference(clean.toPacked(), c.toPacked(), "gemm beta zero")
    }

    @Test
    fun `an empty operation leaves the destination alone`() = withVendor { blas ->
        val a = MatrixWindow(DoubleArray(0), 3, 0)
        val b = MatrixWindow(DoubleArray(0), 0, 5)
        val storage = values(15, 13)
        val original = storage.copyOf()

        blas.gemm(1.0, a, b, 1.0, MatrixWindow(storage, 3, 5))

        assertEquals(original.toList(), storage.toList())
    }

    @Test
    fun `gemv over a submatrix agrees with the reference`() = withVendor { blas ->
        val parent = values(400, 14)
        val a = MatrixWindow(parent, 4, 3, offset = 25, rowStride = 1, columnStride = 30)
        val x = VectorWindow(values(3, 15), 3)
        val storage = values(4, 16)
        val y = VectorWindow(storage, 4)
        val expected = oracleGemv(0.5, a, x, 2.0, y)

        blas.gemv(0.5, a, x, 2.0, y)

        assertAgreesWithReference(expected, y.toPacked(), "gemv")
    }

    @Test
    fun `a symmetric operand reads only its stored triangle`() = withVendor { blas ->
        val order = 5
        val stored = values(order * order, 17)
        for (column in 0 until order) {
            for (row in 0 until column) stored[row + column * order] = Double.NaN
        }
        val a = MatrixWindow(stored, order, order, structure = MatrixStructure.SymmetricLower)
        val x = VectorWindow(values(order, 18), order)
        val y = VectorWindow(DoubleArray(order), order)
        val expected = oracleGemv(1.0, a, x, 0.0, y)

        blas.symv(1.0, a, x, 0.0, y)

        assertAgreesWithReference(expected, y.toPacked(), "symv")
    }

    @Test
    fun `a unit diagonal is never loaded from storage`() = withVendor { blas ->
        val order = 4
        val stored = values(order * order, 19)
        for (index in 0 until order) stored[index + index * order] = Double.NaN
        val a = MatrixWindow(stored, order, order, structure = MatrixStructure.UnitLower)
        val rightHandSide = values(order, 20)
        val x = VectorWindow(rightHandSide.copyOf(), order)

        blas.trsv(a, x)

        val residual = oracleGemv(1.0, a, x, 0.0, VectorWindow(DoubleArray(order), order))
        assertAgreesWithReference(rightHandSide, residual, "trsv unit diagonal")
    }

    @Test
    fun `a triangular solve over a submatrix inverts its own product`() = withVendor { blas ->
        val order = 4
        val parent = values(400, 21)
        val a = MatrixWindow(parent, order, order, offset = 11, rowStride = 1, columnStride = 25)
        for (index in 0 until order) parent[11 + index + index * 25] = 3.0 + index
        val triangular = MatrixWindow(
            parent,
            order,
            order,
            offset = 11,
            rowStride = 1,
            columnStride = 25,
            structure = MatrixStructure.TriangularLower,
        )
        val original = values(order, 22)
        val x = VectorWindow(original.copyOf(), order)

        blas.trsv(triangular, x)
        blas.trmv(triangular, x)

        assertAgreesWithReference(original, x.toPacked(), "trsv then trmv")
        assertTrue(a.rows == order)
    }

    @Test
    fun `syrk fills only the selected triangle`() = withVendor { blas ->
        val order = 4
        val a = MatrixWindow(values(order * 3, 23), order, 3)
        val storage = DoubleArray(order * order) { 7.0 }
        val c = MatrixWindow(storage, order, order, structure = MatrixStructure.SymmetricLower)

        blas.syrk(1.0, a, 0.0, c)

        for (column in 0 until order) {
            for (row in 0 until order) {
                val value = storage[row + column * order]
                if (row < column) {
                    assertEquals(7.0, value, "upper triangle at ($row, $column) was written")
                } else {
                    var sum = 0.0
                    for (k in 0 until 3) sum += a[row, k] * a[column, k]
                    assertTrue(abs(sum - value) <= 1e-12 * maxOf(1.0, abs(sum)), "syrk at ($row, $column)")
                }
            }
        }
    }

    @Test
    fun `level one calls agree with the reference including negative strides`() = withVendor { blas ->
        val size = 6
        val xs = values(2 * size, 24)
        val ys = values(2 * size, 25)
        for (stride in listOf(1, 2, -1, -2)) {
            val offset = if (stride > 0) 0 else (size - 1) * -stride
            val x = VectorWindow(xs, size, offset, stride)
            val y = VectorWindow(ys.copyOf(), size, offset, stride)
            val expectedDot = (0 until size).sumOf { x[it] * y[it] }

            assertTrue(abs(blas.dot(x, y) - expectedDot) <= 1e-12 * maxOf(1.0, abs(expectedDot)), "dot at $stride")

            val largest = (0 until size).maxOf { abs(x[it]) }
            assertEquals(largest, abs(x[blas.iamax(x)]), "iamax at $stride did not land on the largest entry")

            val expectedNorm = kotlin.math.sqrt((0 until size).sumOf { x[it] * x[it] })
            assertTrue(
                abs(blas.nrm2(x) - expectedNorm) <= 1e-12 * maxOf(1.0, expectedNorm),
                "nrm2 at $stride",
            )
            val expectedSum = (0 until size).sumOf { abs(x[it]) }
            assertTrue(abs(blas.asum(x) - expectedSum) <= 1e-12 * maxOf(1.0, expectedSum), "asum at $stride")
        }
    }

    @Test
    fun `a backward window reports the last of several equal maxima`() = withVendor { blas ->
        val storage = doubleArrayOf(1.0, 5.0, 2.0, 5.0, 3.0)

        val forward = blas.iamax(VectorWindow(storage, 5, offset = 0, stride = 1))
        val backward = blas.iamax(VectorWindow(storage, 5, offset = 4, stride = -1))

        // The backward window reads [3, 5, 2, 5, 1], whose equal maxima sit at logical 1 and 3.
        assertEquals(1, forward, "a forward window reports the first largest entry")
        assertEquals(3, backward, "a backward window reports the last of the equal maxima")
    }

    @Test
    fun `axpy writes only through the window it was given`() = withVendor { blas ->
        val size = 4
        val storage = values(20, 26)
        val original = storage.copyOf()
        val x = VectorWindow(values(size, 27), size)
        val y = VectorWindow(storage, size, offset = 3, stride = 4)
        val expected = DoubleArray(size) { y[it] + 0.5 * x[it] }

        blas.axpy(0.5, x, y)

        assertAgreesWithReference(expected, y.toPacked(), "axpy")
        for (index in storage.indices) {
            if ((index - 3) % 4 != 0 || index < 3 || index > 3 + 3 * 4) {
                assertEquals(original[index], storage[index], "storage outside the window changed at $index")
            }
        }
    }

    @Test
    fun `an empty vector call returns the identity without touching storage`() = withVendor { blas ->
        val empty = VectorWindow(DoubleArray(0), 0)

        assertEquals(0.0, blas.dot(empty, empty))
        assertEquals(0.0, blas.nrm2(empty))
        assertEquals(0.0, blas.asum(empty))
        assertEquals(0, blas.iamax(empty))
    }

    @Test
    fun `gemmt writes one triangle whether it is bound or composed`() = withVendor { blas ->
        val order = 4
        val a = MatrixWindow(values(order * 3, 28), order, 3)
        val b = MatrixWindow(values(3 * order, 29), 3, order)
        val storage = DoubleArray(order * order) { 5.0 }
        val c = MatrixWindow(storage, order, order, structure = MatrixStructure.SymmetricLower)
        val full = oracleGemm(1.0, a, b, 0.0, MatrixWindow(DoubleArray(order * order), order, order))

        blas.gemmt(1.0, a, b, 0.0, c)

        for (column in 0 until order) {
            for (row in 0 until order) {
                val value = storage[row + column * order]
                if (row < column) {
                    assertEquals(5.0, value, "upper triangle at ($row, $column) was written")
                } else {
                    val wanted = full[row + column * order]
                    assertTrue(abs(wanted - value) <= 1e-12 * maxOf(1.0, abs(wanted)), "gemmt at ($row, $column)")
                }
            }
        }
    }

    @Test
    fun `the resolved library and version identify what actually ran`() = withVendor { blas ->
        assertTrue(blas.version.isNotEmpty(), "no version string")
        assertTrue(VendorOperation.Gemm in blas.directlyImplemented, "gemm should be directly implemented")
    }

    @Test
    fun `the reported library is the file the symbol came from not the name that was asked for`() {
        for (vendor in Vendor.entries) {
            val blas = openVendorBlas(vendor) ?: continue

            assertTrue(
                blas.libraryPath.startsWith("/"),
                "${vendor.vendorName} reported ${blas.libraryPath}, which is a candidate name rather than a file",
            )
            assertEquals(vendor, blas.vendor, "a library resolved under the wrong vendor")
        }
    }
}
