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
 * Operands a call promises not to read.
 *
 * A zero multiplier contributes nothing, and the contracts say so rather than saying the product happens to
 * be zero. The difference is visible only to an operand that cannot be read at all, which is what these
 * poisoned implementations are: a `Vector` from outside the library may compute its entries, refuse them, or
 * cost something to produce, and a rank update that reads one for a multiplier of zero has broken its word.
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

    /**
     * Preparing a snapshot copies values; using it for a product that contributes nothing must not. The
     * derived transposed orientation is the observable part: a zero multiplier leaves it unbuilt, so the
     * snapshot still costs nothing beyond what it was asked for.
     */
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
