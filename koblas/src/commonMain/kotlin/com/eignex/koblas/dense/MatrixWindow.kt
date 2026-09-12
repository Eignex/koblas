package com.eignex.koblas.dense

/** Logical input to a matrix block. Backing storage stays strongly reachable for the operand's lifetime. */
public sealed class MatrixOperand {
    /** Logical row count, after any baked transpose. */
    public abstract val rows: Int

    /** Logical column count, after any baked transpose. */
    public abstract val columns: Int
    internal abstract val buffer: DoubleArray
    internal abstract fun value(row: Int, column: Int): Double
}

/** Structure of a stored matrix; unstored entries and unit diagonals are never loaded. */
public enum class MatrixStructure {
    /** Every entry is stored. */
    General,

    /** The lower triangle is mirrored. */
    SymmetricLower,

    /** The upper triangle is mirrored. */
    SymmetricUpper,

    /** Entries above the diagonal are positive zero. */
    TriangularLower,

    /** Entries below the diagonal are positive zero. */
    TriangularUpper,

    /** Lower triangular with implicit diagonal ones. */
    UnitLower,

    /** Upper triangular with implicit diagonal ones. */
    UnitUpper,
}

/**
 * Validated two-stride window. A(i, j) addresses [offset] + i * [rowStride] + j * [columnStride].
 * Negative strides are supported; distinct logical entries must address distinct storage. Structured windows
 * start square; [window] can select rectangular panels. [transposed] records an already applied transpose;
 * strides, dimensions and structure describe the resulting logical matrix. Storage is borrowed and mutations require
 * synchronization by the caller.
 */
@Suppress("LongParameterList")
public class MatrixWindow private constructor(
    /** Borrowed backing array. */
    public val data: DoubleArray,
    override val rows: Int,
    override val columns: Int,
    /** First physical entry of the window. */
    public val offset: Int,
    /** Physical distance between adjacent rows. */
    public val rowStride: Int,
    /** Physical distance between adjacent columns. */
    public val columnStride: Int,
    /** Stored structure of the logical input. */
    public val structure: MatrixStructure,
    /** Whether a transpose is already represented by this operand. */
    public val transposed: Boolean,
    /** Row origin minus column origin within the structured parent. */
    public val diagonalOffset: Int,
) : MatrixOperand() {
    /** Creates a full matrix window; structured inputs must be square before taking subwindows. */
    public constructor(
        data: DoubleArray,
        rows: Int,
        columns: Int,
        offset: Int = 0,
        rowStride: Int = 1,
        columnStride: Int = maxOf(1, rows),
        structure: MatrixStructure = MatrixStructure.General,
        transposed: Boolean = false,
    ) : this(data, rows, columns, offset, rowStride, columnStride, structure, transposed, 0) {
        require(structure == MatrixStructure.General || rows == columns) { "structured matrix must be square" }
    }

    init {
        require(rows >= 0 && columns >= 0) { "negative matrix dimensions" }
        require(rowStride != 0 && columnStride != 0) { "matrix strides must be nonzero" }
        val rowReach = maxOf(0, rows - 1).toLong() * rowStride
        val columnReach = maxOf(0, columns - 1).toLong() * columnStride
        requireWindowBounds(data.size, offset, rows == 0 || columns == 0, rowReach, columnReach)
        val rowStep = kotlin.math.abs(rowStride.toLong())
        val columnStep = kotlin.math.abs(columnStride.toLong())
        val divisor = greatestCommonDivisor(rowStep, columnStep)
        require(rows <= columnStep / divisor || columns <= rowStep / divisor) { "matrix entries overlap" }
    }

    override val buffer: DoubleArray get() = data

    internal fun index(row: Int, column: Int): Int =
        (offset.toLong() + row.toLong() * rowStride + column.toLong() * columnStride).toInt()

    override fun value(row: Int, column: Int): Double = when (structure) {
        MatrixStructure.General -> data[index(row, column)]

        MatrixStructure.SymmetricLower -> if (row.toLong() + diagonalOffset >= column) {
            data[index(row, column)]
        } else {
            data[index(column - diagonalOffset, row + diagonalOffset)]
        }

        MatrixStructure.SymmetricUpper -> if (row.toLong() + diagonalOffset <= column) {
            data[index(row, column)]
        } else {
            data[index(column - diagonalOffset, row + diagonalOffset)]
        }

        MatrixStructure.TriangularLower -> if (row.toLong() + diagonalOffset >= column) {
            data[index(row, column)]
        } else {
            0.0
        }

        MatrixStructure.TriangularUpper -> if (row.toLong() + diagonalOffset <= column) {
            data[index(row, column)]
        } else {
            0.0
        }

        MatrixStructure.UnitLower -> when {
            row.toLong() + diagonalOffset == column.toLong() -> 1.0
            row.toLong() + diagonalOffset > column -> data[index(row, column)]
            else -> 0.0
        }

        MatrixStructure.UnitUpper -> when {
            row.toLong() + diagonalOffset == column.toLong() -> 1.0
            row.toLong() + diagonalOffset < column -> data[index(row, column)]
            else -> 0.0
        }
    }

    /** Reads one logical entry, applying the declared structure. */
    public operator fun get(row: Int, column: Int): Double {
        require(row in 0 until rows && column in 0 until columns) { "matrix index out of bounds" }
        return value(row, column)
    }

    /** A zero-copy transpose, including reversal of the selected stored triangle. */
    public fun transpose(): MatrixWindow = MatrixWindow(
        data,
        columns,
        rows,
        offset,
        columnStride,
        rowStride,
        when (structure) {
            MatrixStructure.General -> MatrixStructure.General
            MatrixStructure.SymmetricLower -> MatrixStructure.SymmetricUpper
            MatrixStructure.SymmetricUpper -> MatrixStructure.SymmetricLower
            MatrixStructure.TriangularLower -> MatrixStructure.TriangularUpper
            MatrixStructure.TriangularUpper -> MatrixStructure.TriangularLower
            MatrixStructure.UnitLower -> MatrixStructure.UnitUpper
            MatrixStructure.UnitUpper -> MatrixStructure.UnitLower
        },
        !transposed,
        -diagonalOffset,
    )

    /**
     * A zero-copy logical block. Structured blocks retain parent coordinates, so symmetric mirroring can read
     * the corresponding stored entry outside the block while staying inside the validated parent allocation.
     */
    public fun window(row: Int, rows: Int, column: Int, columns: Int): MatrixWindow {
        require(row >= 0 && rows >= 0 && row.toLong() + rows <= this.rows) { "row window exceeds matrix" }
        require(
            column >= 0 && columns >= 0 && column.toLong() + columns <= this.columns,
        ) { "column window exceeds matrix" }
        val origin = offset.toLong() + row.toLong() * rowStride + column.toLong() * columnStride
        val start = if (rows == 0 || columns == 0) origin.coerceIn(0, data.size.toLong()).toInt() else origin.toInt()
        return MatrixWindow(
            data, rows, columns, start, rowStride, columnStride, structure, transposed,
            (diagonalOffset.toLong() + row - column).toInt(),
        )
    }

    internal fun implicitZero(row: Int, column: Int): Boolean = when (structure) {
        MatrixStructure.TriangularLower, MatrixStructure.UnitLower -> row.toLong() + diagonalOffset < column
        MatrixStructure.TriangularUpper, MatrixStructure.UnitUpper -> row.toLong() + diagonalOffset > column
        else -> false
    }
}

/** Validated borrowed vector window, including negative strides. */
public class VectorWindow(
    /** Borrowed backing array. */
    public val data: DoubleArray,
    /** Logical entry count. */
    public val size: Int,
    /** First physical entry of the window. */
    public val offset: Int = 0,
    /** Physical distance between adjacent entries. */
    public val stride: Int = 1,
) {
    init {
        require(size >= 0 && stride != 0) { "invalid vector size or stride" }
        requireWindowBounds(data.size, offset, size == 0, maxOf(0, size - 1).toLong() * stride, 0)
    }

    /** Reads one logical entry. */
    public operator fun get(index: Int): Double {
        require(index in 0 until size) { "vector index out of bounds" }
        return data[(offset.toLong() + index.toLong() * stride).toInt()]
    }
}

private fun requireWindowBounds(size: Int, offset: Int, empty: Boolean, rowReach: Long, columnReach: Long) {
    if (empty) {
        require(offset in 0..size) { "empty window offset exceeds storage" }
    } else {
        val first = offset.toLong() + minOf(0, rowReach) + minOf(0, columnReach)
        val last = offset.toLong() + maxOf(0, rowReach) + maxOf(0, columnReach)
        require(first >= 0 && last < size) { "window exceeds backing storage" }
    }
}

private fun greatestCommonDivisor(first: Long, second: Long): Long {
    var a = first
    var b = second
    while (b != 0L) {
        val remainder = a % b
        a = b
        b = remainder
    }
    return a
}
