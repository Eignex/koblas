package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG

/**
 * A `cholmod_sparse` over a copy of a koblas matrix, held in [arena] until the caller closes it.
 *
 * The copy is what taking operands as a struct costs. The other bindings hand their arrays straight to the
 * call, where a critical downcall pins them in place; a struct field has to hold an address, and an on-heap
 * array has none to give.
 */
internal class CholmodMatrix private constructor(val segment: MemorySegment, private val arena: Arena) :
    AutoCloseable {

    override fun close(): Unit = arena.close()

    companion object {
        /**
         * [a]'s lower triangle as a `cholmod_sparse`, with `stype` saying so, which is what makes CHOLMOD
         * read that triangle and ignore whatever is stored above the diagonal.
         */
        fun lowerTriangleOf(a: F64SparseMatrix): CholmodMatrix {
            val arena = Arena.ofConfined()
            return try {
                CholmodMatrix(describe(a, arena), arena)
            } catch (@Suppress("TooGenericExceptionCaught") failure: Throwable) {
                // Rethrown, and the arena has no other owner yet, so nothing else would close it.
                arena.close()
                throw failure
            }
        }

        private fun describe(a: F64SparseMatrix, arena: Arena): MemorySegment {
            val columnPointers = arena.allocate(JAVA_INT, a.cols.toLong() + 1)
            MemorySegment.copy(a.colPtr, 0, columnPointers, JAVA_INT, 0L, a.cols + 1)
            val rowIndices = arena.allocate(JAVA_INT, maxOf(a.nnz, 1).toLong())
            MemorySegment.copy(a.rowIdx, 0, rowIndices, JAVA_INT, 0L, a.nnz)
            val values = arena.allocate(JAVA_DOUBLE, maxOf(a.nnz, 1).toLong())
            MemorySegment.copy(a.values, 0, values, JAVA_DOUBLE, 0L, a.nnz)

            val sparse = arena.allocate(CHOLMOD_SPARSE_BYTES)
            sparse.set(JAVA_LONG, CHOLMOD_SPARSE_NROW, a.rows.toLong())
            sparse.set(JAVA_LONG, CHOLMOD_SPARSE_NCOL, a.cols.toLong())
            sparse.set(JAVA_LONG, CHOLMOD_SPARSE_NZMAX, a.nnz.toLong())
            sparse.set(ADDRESS, CHOLMOD_SPARSE_P, columnPointers)
            sparse.set(ADDRESS, CHOLMOD_SPARSE_I, rowIndices)
            // Null because the columns are packed, so their lengths come from the pointers alone.
            sparse.set(ADDRESS, CHOLMOD_SPARSE_NZ, MemorySegment.NULL)
            sparse.set(ADDRESS, CHOLMOD_SPARSE_X, values)
            sparse.set(ADDRESS, CHOLMOD_SPARSE_Z, MemorySegment.NULL)
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_STYPE, CHOLMOD_STYPE_LOWER)
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_ITYPE, CHOLMOD_INT)
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_XTYPE, CHOLMOD_REAL)
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_DTYPE, CHOLMOD_DOUBLE)
            // koblas keeps rows strictly ascending within a column, which is what sorted claims.
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_SORTED, CHOLMOD_TRUE)
            sparse.set(JAVA_INT, CHOLMOD_SPARSE_PACKED, CHOLMOD_TRUE)
            return sparse
        }
    }
}
