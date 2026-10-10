package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Vector
import com.eignex.koblas.gemm
import com.eignex.koblas.gemmInto
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
import com.eignex.koblas.syr
import com.eignex.koblas.syr2
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.fail

/**
 * Zero multipliers must leave operands unread. Poisoned [Vector] implementations make reads
 * observable even when the numerical contribution would be zero.
 */
class SparseNoReadTest {

    /** A [Vector] whose entries cannot be read, only counted. */
    private class PoisonVector(override val size: Int) : Vector {
        override fun get(i: Int): Double = fail("a vector this call must not read was read at $i")
        override fun toDoubleArray(): DoubleArray = fail("a vector this call must not read was materialised")
    }

    private fun example(): SparseMatrix = SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 2.0, 1 to 3.0), listOf(1 to 5.0)),
    )

    @Test
    fun `a zero multiplier rank one update copies the source without reading the vector`() {
        val source = example()

        val result = source.syr(0.0, PoisonVector(2))

        assertEquals(source, result)
    }

    @Test
    fun `a zero multiplier rank two update copies the source without reading either vector`() {
        val source = example()

        val result = source.syr2(0.0, PoisonVector(2), PoisonVector(2))

        assertEquals(source, result)
    }

    @Test
    fun `a rank update result owns its arrays whatever the multiplier`() {
        val source = example()

        val result = source.syr(0.0, PoisonVector(2))
        source.values.fill(Double.NaN)

        assertContentEquals(doubleArrayOf(2.0, 3.0, 5.0), result.values)
    }

    /** A zero multiplier must leave a prepared snapshot's derived transpose unbuilt. */
    @Test
    fun `a zero multiplier prepared product does not derive the transposed orientation`() {
        val prepared = example().prepare()
        val destination = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        prepared.gemmInto(0.0, true, DenseMatrix.diagonal(2), false, 2.0, destination)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0, 8.0), destination.values)
        assertFalse(prepared.orientationDerived, "a product contributing nothing built the transpose cache")
    }

    @Test
    fun `a zero multiplier prepared sparse product discovers structure without the transpose cache`() {
        val prepared = example().prepare()
        val other = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))

        val result = prepared.gemm(0.0, true, other, false)
        val expected = koblas.gemm(0.0, example(), true, other, false)

        assertEquals(expected, result, "the structure a zero multiplier discovers is the same either way")
        assertFalse(prepared.orientationDerived, "a product contributing nothing built the transpose cache")
    }

    @Test
    fun `a prepared product with work to do does derive the transposed orientation once`() {
        val prepared = example().prepare()
        val other = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))

        val first = prepared.gemm(1.0, true, other, false)
        val second = prepared.gemm(1.0, true, other, false)

        assertEquals(first, second)
        assertEquals(koblas.gemm(1.0, example(), true, other, false), first)
        assertEquals(true, prepared.orientationDerived)
    }

    @Test
    fun `a prepared product rejects an impossible shape before deriving anything`() {
        val prepared = example().prepare()
        val mismatched = SparseMatrix.ofColumns(3, 3, List(3) { emptyList() })

        val failure = runCatching { prepared.gemm(1.0, true, mismatched, false) }

        assertEquals(true, failure.isFailure)
        assertFalse(prepared.orientationDerived, "a call rejected for its shape prepared an orientation first")
    }
}
