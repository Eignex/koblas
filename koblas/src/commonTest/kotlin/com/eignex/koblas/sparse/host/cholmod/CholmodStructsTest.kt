package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.withNativeBlocks
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * CHOLMOD takes its operands as structs, so a binding that lays one out wrongly hands the library the wrong
 * matrix without anything raising. Written and read back here, on both platforms, without the library
 * installed: the layout is the part that has to be right before a call is worth making.
 */
class CholmodStructsTest {

    private val matrix = F64SparseMatrix.ofColumns(
        3,
        3,
        listOf(listOf(0 to 4.0, 2 to 1.5), listOf(1 to 3.0), listOf(2 to 2.25)),
    )

    @Test
    fun `a sparse descriptor reads back as the matrix it was built from`() {
        withNativeBlocks { blocks ->
            val sparse = blocks.describeCholmodSparse(matrix, CHOLMOD_STYPE_LOWER)

            assertEquals(matrix, sparse.readCholmodSparse())
        }
    }

    @Test
    fun `a sparse descriptor carries the shape and the codes CHOLMOD reads`() {
        withNativeBlocks { blocks ->
            val sparse = blocks.describeCholmodSparse(matrix, CHOLMOD_STYPE_LOWER)

            assertEquals(3L, sparse.getSize(CHOLMOD_SPARSE_NROW), "nrow")
            assertEquals(3L, sparse.getSize(CHOLMOD_SPARSE_NCOL), "ncol")
            assertEquals(4L, sparse.getSize(CHOLMOD_SPARSE_NZMAX), "nzmax")
            assertEquals(CHOLMOD_STYPE_LOWER, sparse.getInt(CHOLMOD_SPARSE_STYPE), "stype")
            assertEquals(CHOLMOD_INT, sparse.getInt(CHOLMOD_SPARSE_ITYPE), "itype")
            assertEquals(CHOLMOD_REAL, sparse.getInt(CHOLMOD_SPARSE_XTYPE), "xtype")
            assertEquals(CHOLMOD_DOUBLE, sparse.getInt(CHOLMOD_SPARSE_DTYPE), "dtype")
            assertEquals(CHOLMOD_TRUE, sparse.getInt(CHOLMOD_SPARSE_SORTED), "sorted")
            assertEquals(CHOLMOD_TRUE, sparse.getInt(CHOLMOD_SPARSE_PACKED), "packed")
        }
    }

    /** The columns are packed, so a length array would contradict the pointers; CHOLMOD wants it null. */
    @Test
    fun `a packed sparse descriptor leaves its column length pointer null`() {
        withNativeBlocks { blocks ->
            val sparse = blocks.describeCholmodSparse(matrix, CHOLMOD_STYPE_GENERAL)

            assertEquals(null, sparse.getPointer(CHOLMOD_SPARSE_NZ, 0L), "nz")
            assertEquals(null, sparse.getPointer(CHOLMOD_SPARSE_Z, 0L), "z")
        }
    }

    /** An empty pattern has no entries to point at, and the arrays are still allocated for a library to read. */
    @Test
    fun `an empty matrix describes and reads back as empty`() {
        val empty = F64SparseMatrix.ofColumns(2, 2, listOf(emptyList(), emptyList()))

        withNativeBlocks { blocks ->
            val sparse = blocks.describeCholmodSparse(empty, CHOLMOD_STYPE_GENERAL)
            val read = sparse.readCholmodSparse()

            assertEquals(0L, sparse.getSize(CHOLMOD_SPARSE_NZMAX), "nzmax")
            assertEquals(0, read.nnz, "nnz")
            assertEquals(empty, read)
        }
    }

    @Test
    fun `a dense descriptor carries the shape a column-major block has`() {
        withNativeBlocks { blocks ->
            val values = blocks.doubles(6, DoubleArray(6) { it.toDouble() }, 6)

            val dense = blocks.block(CHOLMOD_DENSE_BYTES).asCholmodDense(values, rows = 3, columns = 2)

            assertEquals(3L, dense.getSize(CHOLMOD_DENSE_NROW), "nrow")
            assertEquals(2L, dense.getSize(CHOLMOD_DENSE_NCOL), "ncol")
            assertEquals(6L, dense.getSize(CHOLMOD_DENSE_NZMAX), "nzmax")
            assertEquals(3L, dense.getSize(CHOLMOD_DENSE_D), "leading dimension")
            assertEquals(CHOLMOD_REAL, dense.getInt(CHOLMOD_DENSE_XTYPE), "xtype")
            assertEquals(CHOLMOD_DOUBLE, dense.getInt(CHOLMOD_DENSE_DTYPE), "dtype")
            val held = dense.getPointer(CHOLMOD_DENSE_X, 6L * Double.SIZE_BYTES)
            assertContentEquals(DoubleArray(6) { it.toDouble() }, held?.readDoubles(6), "values")
        }
    }

    /** A library that chose no ordering leaves the pointer null, which every caller reads as the identity. */
    @Test
    fun `an absent permutation reads back as the identity`() {
        withNativeBlocks { blocks ->
            val descriptor = blocks.block(CHOLMOD_FACTOR_BYTES)
            descriptor.putPointer(CHOLMOD_FACTOR_PERM, null)

            assertContentEquals(intArrayOf(0, 1, 2, 3), descriptor.readCholmodPermutation(CHOLMOD_FACTOR_PERM, 4))
        }
    }

    @Test
    fun `a stored permutation reads back in order`() {
        withNativeBlocks { blocks ->
            val descriptor = blocks.block(CHOLMOD_FACTOR_BYTES)
            descriptor.putPointer(CHOLMOD_FACTOR_PERM, blocks.ints(3, intArrayOf(2, 0, 1), 3))

            assertContentEquals(intArrayOf(2, 0, 1), descriptor.readCholmodPermutation(CHOLMOD_FACTOR_PERM, 3))
        }
    }
}
