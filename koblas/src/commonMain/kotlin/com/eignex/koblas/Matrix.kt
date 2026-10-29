package com.eignex.koblas

import kotlinx.serialization.Serializable

/** Read-only matrix contract. Anything that only reads a matrix should take this. */
public interface Matrix {
    /** Number of rows. */
    public val rows: Int

    /** Number of columns. */
    public val cols: Int

    /** The entry at row (i), column (j). Throws `IndexOutOfBoundsException` outside the shape, whatever the storage. */
    public operator fun get(i: Int, j: Int): Double

    /** Materialise into a fresh `Array<DoubleArray>` of rows, independent of the internal storage. */
    public fun toArray(): Array<DoubleArray>
}

/**
 * The matrix storages koblas itself defines, [DenseMatrix] and [SparseMatrix].
 *
 * Encoded on the terms [VectorStorage] states: the storage's own class name, the numbers under `values`, and
 * each index array named for the axis it indexes.
 */
@Serializable
public sealed interface MatrixStorage : Matrix
