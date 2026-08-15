package com.eignex.koblas

import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** Indexing and shape validation, which must not depend on which storage the caller happens to hold. */
class BoundsAndShapeTest {

    private val dense = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
    private val sparse = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 5.0), listOf()))

    @Test
    fun `every matrix storage rejects an index outside the shape`() {
        for (a in listOf<MatrixLike>(dense, sparse)) {
            val what = a::class.simpleName
            assertFailsWith<IndexOutOfBoundsException>("$what row at rows") { a[2, 0] }
            assertFailsWith<IndexOutOfBoundsException>("$what row beyond rows") { a[5, 0] }
            assertFailsWith<IndexOutOfBoundsException>("$what negative row") { a[-1, 1] }
            assertFailsWith<IndexOutOfBoundsException>("$what col at cols") { a[0, 3] }
            assertFailsWith<IndexOutOfBoundsException>("$what negative col") { a[0, -1] }
        }
    }

    @Test
    fun `every vector storage rejects an index outside the size`() {
        val vectors = listOf<VectorLike>(
            DenseVector.of(doubleArrayOf(1.0, 2.0)),
            SparseVector.of(2, intArrayOf(0), doubleArrayOf(1.0)),
        )
        for (v in vectors) {
            val what = v::class.simpleName
            assertFailsWith<IndexOutOfBoundsException>("$what at size") { v[2] }
            assertFailsWith<IndexOutOfBoundsException>("$what beyond size") { v[7] }
            assertFailsWith<IndexOutOfBoundsException>("$what negative") { v[-1] }
        }
    }

    @Test
    fun `an out-of-range row does not read a neighbouring column`() {
        // data is column-major, so row 5 of a 2x3 lands inside column 2 and used to read back as 6.0.
        assertFailsWith<IndexOutOfBoundsException> { dense[5, 0] }
        assertFailsWith<IndexOutOfBoundsException> { dense[5, 0] = 0.0 }
        assertEquals(6.0, dense[1, 2], "the entry that used to leak must still be reachable properly")
    }

    @Test
    fun `a shape whose entry count overflows an Int is rejected`() {
        assertFailsWith<DimensionMismatch>("wraps to exactly zero") { DenseMatrix.zero(65536, 65536) }
        assertFailsWith<DimensionMismatch>("wraps negative") { DenseMatrix.zero(46341, 46341) }
        assertFailsWith<DimensionMismatch>("one huge dimension") { DenseMatrix.zero(2, 1073741824) }
        assertEquals(4, DenseMatrix.zero(2, 2).data.size, "an ordinary shape still works")
    }

    @Test
    fun `a negative shape is rejected as a shape error`() {
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(-1, 2) }
        assertFailsWith<DimensionMismatch> { DenseMatrix.zero(2, -1) }
        assertFailsWith<DimensionMismatch> { DenseVector.zero(-3) }
        assertFailsWith<DimensionMismatch> { SparseVector.wrap(-5, IntArray(0), DoubleArray(0)) }
        assertFailsWith<DimensionMismatch> { SparseVector.of(-3, IntArray(0), DoubleArray(0)) }
        assertFailsWith<DimensionMismatch> { SparseMatrix.ofTriplets(-1, 1, IntArray(0), IntArray(0), DoubleArray(0)) }
    }

    @Test
    fun `a negative size cannot arrive through deserialization either`() {
        assertFailsWith<DimensionMismatch> {
            Json.decodeFromString<VectorView>("""{"type":"SparseVector","size":-5,"indices":[],"values":[]}""")
        }
    }
}
