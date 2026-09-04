package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG

/*
 * Reading CSC out of a CHOLMOD descriptor, which SuiteSparse hands back from more than one call. A sparse
 * matrix and a factor keep their arrays at different offsets and are read the same way, so what differs
 * between the callers is three offsets rather than the six steps.
 */

/** The three arrays behind a descriptor, before anything decides what matrix they describe. */
internal class CholmodCsc(val colPtr: IntArray, val rowIdx: IntArray, val values: DoubleArray)

/**
 * The CSC arrays [descriptor] points at, with [pointers], [indices] and [entries] naming the fields to read
 * them from. An empty pattern copies neither the indices nor the values, since a library with nothing to
 * report may leave those pointers null.
 */
internal fun readCholmodCsc(
    descriptor: MemorySegment,
    cols: Int,
    pointers: Long,
    indices: Long,
    entries: Long,
): CholmodCsc {
    val colPtr = IntArray(cols + 1)
    val p = descriptor.get(ADDRESS, pointers).reinterpret((cols + 1L) * Int.SIZE_BYTES)
    MemorySegment.copy(p, JAVA_INT, 0L, colPtr, 0, colPtr.size)
    val nnz = colPtr[cols]
    val rowIdx = IntArray(nnz)
    val values = DoubleArray(nnz)
    if (nnz > 0) {
        val i = descriptor.get(ADDRESS, indices).reinterpret(nnz.toLong() * Int.SIZE_BYTES)
        val x = descriptor.get(ADDRESS, entries).reinterpret(nnz.toLong() * Double.SIZE_BYTES)
        MemorySegment.copy(i, JAVA_INT, 0L, rowIdx, 0, nnz)
        MemorySegment.copy(x, JAVA_DOUBLE, 0L, values, 0, nnz)
    }
    return CholmodCsc(colPtr, rowIdx, values)
}

/** A `cholmod_sparse` copied out into a matrix of koblas's own. */
internal fun readCholmodSparse(sparse: MemorySegment): F64SparseMatrix {
    val descriptor = sparse.reinterpret(CHOLMOD_SPARSE_BYTES)
    val rows = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NROW).toInt()
    val cols = descriptor.get(JAVA_LONG, CHOLMOD_SPARSE_NCOL).toInt()
    val csc = readCholmodCsc(descriptor, cols, CHOLMOD_SPARSE_P, CHOLMOD_SPARSE_I, CHOLMOD_SPARSE_X)
    return F64SparseMatrix.wrap(rows, cols, csc.colPtr, csc.rowIdx, csc.values)
}

/** A fill-reducing ordering of [size], or the identity where the library left the pointer null. */
internal fun readCholmodPermutation(pointer: MemorySegment, size: Int): IntArray {
    if (pointer.address() == 0L) return IntArray(size) { it }
    val out = IntArray(size)
    MemorySegment.copy(pointer.reinterpret(size.toLong() * Int.SIZE_BYTES), JAVA_INT, 0L, out, 0, size)
    return out
}
