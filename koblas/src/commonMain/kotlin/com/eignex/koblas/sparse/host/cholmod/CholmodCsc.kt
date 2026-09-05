@file:OptIn(UnsafeKoblasApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.internal.host.NativeBlockAllocator

/*
 * CHOLMOD's structs, written and read once for both bindings.
 *
 * The offsets beside these in CholmodAbi.kt were already shared; the field-by-field marshalling was not, and
 * a layout written down in two places is a layout that can be corrected in one of them. Only allocation
 * stays platform specific, which is what the caller's [NativeBlockAllocator] carries.
 */

/** The three arrays behind a CSC descriptor, before anything decides what matrix they describe. */
internal class CholmodCsc(val colPtr: IntArray, val rowIdx: IntArray, val values: DoubleArray)

/**
 * [a] as a `cholmod_sparse` over a copy of its arrays, with [stype] saying which triangle CHOLMOD should
 * read.
 *
 * The copy is what taking operands as a struct costs: a struct field holds an address, and a Kotlin array
 * has none to leave behind once the call that pinned it returns.
 */
internal fun NativeBlockAllocator.describeCholmodSparse(a: F64SparseMatrix, stype: Int): NativeBlock {
    val capacity = maxOf(a.nnz, 1)
    val columnPointers = ints(a.cols + 1, a.colPtr, a.cols + 1)
    val rowIndices = ints(capacity, a.rowIdx, a.nnz)
    val values = doubles(capacity, a.values, a.nnz)

    val sparse = block(CHOLMOD_SPARSE_BYTES)
    sparse.putSize(CHOLMOD_SPARSE_NROW, a.rows.toLong())
    sparse.putSize(CHOLMOD_SPARSE_NCOL, a.cols.toLong())
    sparse.putSize(CHOLMOD_SPARSE_NZMAX, a.nnz.toLong())
    sparse.putPointer(CHOLMOD_SPARSE_P, columnPointers)
    sparse.putPointer(CHOLMOD_SPARSE_I, rowIndices)
    // Null because the columns are packed, so their lengths come from the pointers alone.
    sparse.putPointer(CHOLMOD_SPARSE_NZ, null)
    sparse.putPointer(CHOLMOD_SPARSE_X, values)
    sparse.putPointer(CHOLMOD_SPARSE_Z, null)
    sparse.putInt(CHOLMOD_SPARSE_STYPE, stype)
    sparse.putInt(CHOLMOD_SPARSE_ITYPE, CHOLMOD_INT)
    sparse.putInt(CHOLMOD_SPARSE_XTYPE, CHOLMOD_REAL)
    sparse.putInt(CHOLMOD_SPARSE_DTYPE, CHOLMOD_DOUBLE)
    // koblas keeps rows strictly ascending within a column, which is what sorted claims.
    sparse.putInt(CHOLMOD_SPARSE_SORTED, CHOLMOD_TRUE)
    sparse.putInt(CHOLMOD_SPARSE_PACKED, CHOLMOD_TRUE)
    return sparse
}

/**
 * Fills this block as the `cholmod_dense` of [rows] by [columns] real entries held column-major in [values],
 * and answers with it.
 *
 * The block is the caller's because its lifetime is: a solve keeps one descriptor for as long as the factor
 * lives, where a product builds one per call.
 */
internal fun NativeBlock.asCholmodDense(values: NativeBlock, rows: Int, columns: Int): NativeBlock {
    putSize(CHOLMOD_DENSE_NROW, rows.toLong())
    putSize(CHOLMOD_DENSE_NCOL, columns.toLong())
    putSize(CHOLMOD_DENSE_NZMAX, rows.toLong() * columns)
    // The leading dimension: the columns are contiguous, so a column starts every `rows` entries.
    putSize(CHOLMOD_DENSE_D, rows.toLong())
    putPointer(CHOLMOD_DENSE_X, values)
    putPointer(CHOLMOD_DENSE_Z, null)
    putInt(CHOLMOD_DENSE_XTYPE, CHOLMOD_REAL)
    putInt(CHOLMOD_DENSE_DTYPE, CHOLMOD_DOUBLE)
    return this
}

/**
 * The CSC arrays this descriptor points at, with [pointers], [indices] and [entries] naming the fields to
 * read them from. A `cholmod_sparse` and a `cholmod_factor` keep theirs at different offsets and are read
 * the same way, so what differs between the callers is three offsets rather than the six steps.
 *
 * An empty pattern copies neither the indices nor the values, since a library with nothing to report may
 * leave those pointers null.
 */
internal fun NativeBlock.readCholmodCsc(cols: Int, pointers: Long, indices: Long, entries: Long): CholmodCsc {
    val p = checkNotNull(getPointer(pointers, (cols + 1L) * Int.SIZE_BYTES)) { "CHOLMOD left its column pointers null" }
    val colPtr = p.readInts(cols + 1)
    val nnz = colPtr[cols]
    if (nnz == 0) return CholmodCsc(colPtr, IntArray(0), DoubleArray(0))
    val i = checkNotNull(getPointer(indices, nnz.toLong() * Int.SIZE_BYTES)) { "CHOLMOD left its row indices null" }
    val x = checkNotNull(getPointer(entries, nnz.toLong() * Double.SIZE_BYTES)) { "CHOLMOD left its values null" }
    return CholmodCsc(colPtr, i.readInts(nnz), x.readDoubles(nnz))
}

/** A `cholmod_sparse` copied out into a matrix of koblas's own. */
internal fun NativeBlock.readCholmodSparse(): F64SparseMatrix {
    val descriptor = sized(CHOLMOD_SPARSE_BYTES)
    val rows = descriptor.getSize(CHOLMOD_SPARSE_NROW).toInt()
    val cols = descriptor.getSize(CHOLMOD_SPARSE_NCOL).toInt()
    val csc = descriptor.readCholmodCsc(cols, CHOLMOD_SPARSE_P, CHOLMOD_SPARSE_I, CHOLMOD_SPARSE_X)
    return F64SparseMatrix.wrap(rows, cols, csc.colPtr, csc.rowIdx, csc.values)
}

/** The fill-reducing ordering a factor of [size] was built under, or the identity where it holds none. */
internal fun NativeBlock.readCholmodPermutation(field: Long, size: Int): IntArray {
    val perm = getPointer(field, size.toLong() * Int.SIZE_BYTES) ?: return IntArray(size) { it }
    return perm.readInts(size)
}
