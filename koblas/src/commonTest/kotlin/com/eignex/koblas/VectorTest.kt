package com.eignex.koblas

import kotlin.test.*

class VectorTest {
    @Test
    fun `vector indexing rejects every address outside its size`() {
        val vectors = listOf<Vector>(
            DenseVector.of(doubleArrayOf(1.0, 2.0)),
            SparseVector.of(2, intArrayOf(0), doubleArrayOf(1.0)),
        )
        for (vector in vectors) {
            assertFailsWith<IndexOutOfBoundsException> { vector[2] }
            assertFailsWith<IndexOutOfBoundsException> { vector[7] }
            assertFailsWith<IndexOutOfBoundsException> { vector[-1] }
        }
    }
}
