package com.eignex.koblas

import kotlinx.serialization.json.Json
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

    @Test
    fun `VectorStorage round trips polymorphically preserving dense and sparse types`() {
        val dense: VectorStorage = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val sparse: VectorStorage = SparseVector.of(4, intArrayOf(0, 3), doubleArrayOf(1.0, 2.0))
        val encodedDense = Json.encodeToString(VectorStorage.serializer(), dense)
        val encodedSparse = Json.encodeToString(VectorStorage.serializer(), sparse)
        val backDense = Json.decodeFromString(VectorStorage.serializer(), encodedDense)
        val backSparse = Json.decodeFromString(VectorStorage.serializer(), encodedSparse)

        // Concrete storage names let the payload decode into the same container type.
        for ((storage, encoded) in listOf(dense to encodedDense, sparse to encodedSparse)) {
            assertTrue(storage::class.simpleName!! in encoded, "expected a type discriminator in $encoded")
        }
        assertTrue(backDense is DenseVector)
        assertTrue(backSparse is SparseVector)
        assertEquals(dense, backDense)
        assertEquals(sparse, backSparse)
    }
}
