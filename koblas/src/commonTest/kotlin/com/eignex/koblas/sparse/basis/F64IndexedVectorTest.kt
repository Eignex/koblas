package com.eignex.koblas.sparse.basis

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.*

@OptIn(UnsafeKoblasApi::class)
class F64IndexedVectorTest {
    @Test
    fun `a stored entry is readable and counted`() {
        val x = F64IndexedVector(5)

        x.store(3, 2.5)

        assertEquals(2.5, x[3])
        assertEquals(1, x.count)
    }

    @Test
    fun `clear empties the index set and zeros what was stored`() {
        val x = F64IndexedVector(4)
        x.store(1, 7.0)
        x.store(3, -1.0)

        x.clear()

        assertEquals(0, x.count)
        assertEquals(0.0, x[1])
        assertEquals(0.0, x[3])
    }

    @Test
    fun `unit holds the corresponding basis vector`() {
        val x = F64IndexedVector(3)
        x.store(0, 9.0)

        x.unit(2)

        assertContentEquals(doubleArrayOf(0.0, 0.0, 1.0), x.toDoubleArray())
    }

    @Test
    fun `scatterColumn takes the stored entries of that column`() {
        val a = F64SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0), listOf(1 to 4.0, 2 to 5.0)))
        val x = F64IndexedVector(3)

        x.scatterColumn(a, 1)

        assertEquals(2, x.count)
        assertContentEquals(doubleArrayOf(0.0, 4.0, 5.0), x.toDoubleArray())
    }

    @Test
    fun `scatter indexes only the nonzeros of a dense vector`() {
        val x = F64IndexedVector(4)

        x.scatter(doubleArrayOf(0.0, 3.0, 0.0, -2.0))

        assertEquals(2, x.count)
        assertEquals(0.5, x.density)
    }

    @Test
    fun `gather writes the vector densely`() {
        val x = F64IndexedVector(3)
        x.store(2, 6.0)

        assertContentEquals(doubleArrayOf(0.0, 0.0, 6.0), x.gather(DoubleArray(3)))
    }

    @Test
    fun `tighten drops the entries at or below the tolerance`() {
        val x = F64IndexedVector(4)
        x.store(0, 1.0)
        x.store(1, 1e-14)
        x.store(2, -3.0)

        x.tighten(1e-12)

        assertEquals(2, x.count)
        assertEquals(0.0, x[1])
    }

    @Test
    fun `an index outlives its value cancelling`() {
        val x = F64IndexedVector(3)
        x.store(1, 4.0)

        x.tighten(0.0)

        assertEquals(1, x.count)
    }

    @Test
    fun `reindex recovers the set after a write past the seam`() {
        val x = F64IndexedVector(4)
        x.values[2] = 5.0

        x.reindex()

        assertEquals(1, x.count)
        assertEquals(5.0, x[2])
    }

    @Test
    fun `scatter rejects a dense vector of another length`() {
        val x = F64IndexedVector(3)

        assertFailsWith<DimensionMismatch> { x.scatter(DoubleArray(4)) }
    }

    @Test
    fun `storing outside the vector is rejected`() {
        val x = F64IndexedVector(3)

        assertFailsWith<IndexOutOfBoundsException> { x.store(3, 1.0) }
    }
}
