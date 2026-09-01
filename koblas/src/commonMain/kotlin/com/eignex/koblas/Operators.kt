package com.eignex.koblas

import com.eignex.koblas.core.*

/** `A * B` (BLAS `dgemm`), allocating. gemm accumulates into an existing C instead. */
public operator fun F64DenseMatrix.times(other: F64DenseMatrix): F64DenseMatrix = koblas.gemm(this, other)

/** `A * B` for a sparse left operand and dense right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun F64SparseMatrix.times(other: F64DenseMatrix): F64DenseMatrix = koblas.sparseBlas.gemm(this, other)

/** `A * B` for a dense left operand and sparse right operand, allocating dense storage through the active
 *  sparse backend. */
public operator fun F64DenseMatrix.times(other: F64SparseMatrix): F64DenseMatrix {
    val out = F64DenseMatrix.zero(rows, other.cols)
    koblas.sparseBlas.gemm(1.0, other, false, this, false, 0.0, out, right = true)
    return out
}

/** `A * B` for sparse matrices, allocating the discovered sparse structure through the active backend. */
public operator fun F64SparseMatrix.times(other: F64SparseMatrix): F64SparseMatrix = koblas.sparseBlas.gemm(this, other)

/**
 * Matrix-vector product into a fresh dense result for any [F64MatrixLike] against any [F64VectorLike].
 * gemv provide transpose and destination-buffer variants.
 */
public operator fun F64MatrixLike.times(x: F64VectorLike): F64DenseVector {
    val a = this
    requireShape(a.cols == x.size) { "times shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    if (a is F64DenseMatrix && x is F64DenseVector) return F64DenseVector.wrap(koblas.gemv(a, x.data))
    if (a is F64SparseMatrix && x is F64DenseVector) return F64DenseVector.wrap(koblas.gemv(a, x.data))
    val out = F64DenseVector(a.rows)
    val od = out.data
    when (a) {
        is F64DenseMatrix -> {
            val ad = a.data
            val rows = a.rows
            x.forEachStored { j, v ->
                if (v != 0.0) koblas.kernels.axpy(od, 0, v, ad, j * rows, rows)
            }
        }

        is F64SparseMatrix -> {
            x.forEachStored { j, v ->
                if (v != 0.0) a.forEachInColumn(j) { i, aij -> od[i] += aij * v }
            }
        }

        else -> {
            for (i in 0 until a.rows) {
                var sum = 0.0
                x.forEachStored { j, v -> sum += a[i, j] * v }
                od[i] = sum
            }
        }
    }
    return out
}

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun F64DenseMatrix.times(alpha: Double): F64DenseMatrix = F64DenseMatrix.wrap(
    rows,
    cols,
    scaledCopy(data, alpha),
)

/** `alpha * A`, allocating. [scale] multiplies in place. */
public operator fun Double.times(a: F64DenseMatrix): F64DenseMatrix = a * this

/** `-A`, allocating. */
public operator fun F64DenseMatrix.unaryMinus(): F64DenseMatrix = this * -1.0

/** `A + B`, allocating. [axpy] accumulates into an existing operand. */
public operator fun F64DenseMatrix.plus(other: F64DenseMatrix): F64DenseMatrix = combine(other, 1.0, "plus")

/** `A - B`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun F64DenseMatrix.minus(other: F64DenseMatrix): F64DenseMatrix = combine(other, -1.0, "minus")

/** `a + b`, allocating. [axpy] accumulates into an existing operand. */
public operator fun F64DenseVector.plus(other: F64DenseVector): F64DenseVector = combine(other, 1.0, "plus")

/** `a - b`, allocating. [axpy] with `alpha = -1.0` accumulates into an existing operand. */
public operator fun F64DenseVector.minus(other: F64DenseVector): F64DenseVector = combine(other, -1.0, "minus")

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun F64DenseVector.times(alpha: Double): F64DenseVector = F64DenseVector.wrap(scaledCopy(data, alpha))

/** `alpha * x`, allocating. [scale] multiplies in place. */
public operator fun Double.times(x: F64DenseVector): F64DenseVector = x * this

/** `-x`, allocating. */
public operator fun F64DenseVector.unaryMinus(): F64DenseVector = this * -1.0

/**
 * `alpha * A`, allocating. The CSC pattern carries over untouched, explicitly stored zeros included, so a
 * zero [alpha] returns a matrix of stored zeros rather than an empty one. [scaleColumns] multiplies in place.
 */
public operator fun F64SparseMatrix.times(alpha: Double): F64SparseMatrix = F64SparseMatrix.wrap(
    rows,
    cols,
    copyColumnPointers(),
    copyRowIndices(),
    scaledCopy(values, alpha),
)

/** `alpha * A`, allocating; see [F64SparseMatrix.times]. */
public operator fun Double.times(a: F64SparseMatrix): F64SparseMatrix = a * this

/** `-A`, allocating, over the same CSC pattern. */
public operator fun F64SparseMatrix.unaryMinus(): F64SparseMatrix = this * -1.0

/** `A + B`, allocating, over the union of the two CSC patterns; see [combine] for what the union stores. */
public operator fun F64SparseMatrix.plus(other: F64SparseMatrix): F64SparseMatrix = combine(other, 1.0, "plus")

/** `A - B`, allocating, over the union of the two CSC patterns; see [combine] for what the union stores. */
public operator fun F64SparseMatrix.minus(other: F64SparseMatrix): F64SparseMatrix = combine(other, -1.0, "minus")

/** `alpha * x`, allocating. The stored pattern carries over untouched. [scale] multiplies in place. */
public operator fun F64SparseVector.times(alpha: Double): F64SparseVector =
    F64SparseVector(size, indices.copyOf(), scaledCopy(values, alpha))

/** `alpha * x`, allocating; see [F64SparseVector.times]. */
public operator fun Double.times(x: F64SparseVector): F64SparseVector = x * this

/** `-x`, allocating, over the same stored pattern. */
public operator fun F64SparseVector.unaryMinus(): F64SparseVector = this * -1.0

/** `x + y`, allocating, over the union of the two stored patterns; see [combine]. */
public operator fun F64SparseVector.plus(other: F64SparseVector): F64SparseVector = combine(other, 1.0, "plus")

/** `x - y`, allocating, over the union of the two stored patterns; see [combine]. */
public operator fun F64SparseVector.minus(other: F64SparseVector): F64SparseVector = combine(other, -1.0, "minus")

/**
 * `A + alpha * B` over the union of the two CSC patterns.
 *
 * The union is structural: a position either side stores is stored in the result, even where the two values
 * cancel to zero, so the pattern never depends on the numbers. That matches the rest of the sparse surface,
 * where a fresh structural result keeps discovered fill rather than pruning it.
 */
private fun F64SparseMatrix.combine(other: F64SparseMatrix, alpha: Double, op: String): F64SparseMatrix {
    requireShape(rows == other.rows && cols == other.cols) {
        "$op shape mismatch: ${rows}x$cols and ${other.rows}x${other.cols}"
    }
    val out = SparseUnionBuilder(nnz + other.nnz)
    val pointers = IntArray(cols + 1)
    for (j in 0 until cols) {
        pointers[j] = out.size
        var mine = colPtr[j]
        var theirs = other.colPtr[j]
        while (mine < colPtr[j + 1] || theirs < other.colPtr[j + 1]) {
            val myRow = if (mine < colPtr[j + 1]) rowIdx[mine] else Int.MAX_VALUE
            val theirRow = if (theirs < other.colPtr[j + 1]) other.rowIdx[theirs] else Int.MAX_VALUE
            when {
                myRow < theirRow -> out.add(myRow, values[mine++])
                theirRow < myRow -> out.add(theirRow, alpha * other.values[theirs++])
                else -> out.add(myRow, values[mine++] + alpha * other.values[theirs++])
            }
        }
    }
    pointers[cols] = out.size
    return F64SparseMatrix.wrap(rows, cols, pointers, out.indices(), out.values())
}

/** `x + alpha * y` over the union of the two stored patterns, storing a cancellation as [combine] does. */
private fun F64SparseVector.combine(other: F64SparseVector, alpha: Double, op: String): F64SparseVector {
    requireShape(size == other.size) { "$op size mismatch: $size vs ${other.size}" }
    val out = SparseUnionBuilder(values.size + other.values.size)
    var mine = 0
    var theirs = 0
    while (mine < indices.size || theirs < other.indices.size) {
        val myIndex = if (mine < indices.size) indices[mine] else Int.MAX_VALUE
        val theirIndex = if (theirs < other.indices.size) other.indices[theirs] else Int.MAX_VALUE
        when {
            myIndex < theirIndex -> out.add(myIndex, values[mine++])
            theirIndex < myIndex -> out.add(theirIndex, alpha * other.values[theirs++])
            else -> out.add(myIndex, values[mine++] + alpha * other.values[theirs++])
        }
    }
    return F64SparseVector(size, out.indices(), out.values())
}

/** Collects the merged entries of a sparse union, sized up front from the two operands' stored counts. */
private class SparseUnionBuilder(capacity: Int) {
    private val positions = IntArray(capacity)
    private val coefficients = DoubleArray(capacity)
    var size: Int = 0
        private set

    fun add(position: Int, value: Double) {
        positions[size] = position
        coefficients[size] = value
        size++
    }

    fun indices(): IntArray = positions.copyOf(size)

    fun values(): DoubleArray = coefficients.copyOf(size)
}

/** `A + alpha * B` as a single `axpy` over the flat backings. */
private fun F64DenseMatrix.combine(other: F64DenseMatrix, alpha: Double, op: String): F64DenseMatrix {
    requireShape(rows == other.rows && cols == other.cols) {
        "$op shape mismatch: ${rows}x$cols and ${other.rows}x${other.cols}"
    }
    return F64DenseMatrix.wrap(rows, cols, axpyCopy(data, alpha, other.data))
}

/** `x + alpha * y`. */
private fun F64DenseVector.combine(other: F64DenseVector, alpha: Double, op: String): F64DenseVector {
    requireShape(size == other.size) { "$op size mismatch: $size vs ${other.size}" }
    return F64DenseVector.wrap(axpyCopy(data, alpha, other.data))
}

/** A fresh copy of [data] scaled by [alpha], which is what the allocating scalar products all return. */
private fun scaledCopy(data: DoubleArray, alpha: Double): DoubleArray {
    val out = data.copyOf()
    koblas.kernels.scale(out, 0, alpha, out.size)
    return out
}

/** A fresh copy of [a] with `alpha * b` added, which is what the allocating sums and differences return. */
private fun axpyCopy(a: DoubleArray, alpha: Double, b: DoubleArray): DoubleArray {
    val out = a.copyOf()
    koblas.kernels.axpy(out, 0, alpha, b, 0, out.size)
    return out
}
