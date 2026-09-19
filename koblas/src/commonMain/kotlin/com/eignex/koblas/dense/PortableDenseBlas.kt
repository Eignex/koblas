@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
    "MaxLineLength",
)

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.vendor.*

/** Common Kotlin dense BLAS used by every built-in engine without requiring a host library. */
internal class PortableDenseBlas : DenseBlas {
    private fun scaled(beta: Double, previous: Double): Double = if (beta == 0.0) 0.0 else beta * previous

    override fun gemv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, transpose: Boolean) {
        requireGemvOperands(a, transpose, x.asVector(), y.asVector())
        if (alpha == 0.0 || (if (transpose) a.rows else a.cols) == 0) {
            for (i in y.indices) y[i] = scaled(beta, y[i])
            return
        }
        val stableA = if (a.values === y) a.values.copyOf() else a.values
        val stableX = if (x === y) x.copyOf() else x
        val m = if (transpose) a.cols else a.rows
        val k = if (transpose) a.rows else a.cols
        for (i in 0 until m) {
            var sum = 0.0
            for (p in 0 until k) {
                val av = if (transpose) stableA[p + i * a.rows] else stableA[i + p * a.rows]
                sum += av * stableX[p]
            }
            y[i] = alpha * sum + scaled(beta, y[i])
        }
    }

    override fun transpose(a: DenseMatrix): DenseMatrix {
        val result = DenseMatrix.zero(a.cols, a.rows)
        for (j in 0 until a.cols) for (i in 0 until a.rows) result.values[j + i * a.cols] = a.values[i + j * a.rows]
        return result
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scale(c.values, beta)
            return
        }
        val av = if (a.values === c.values) a.values.copyOf() else a.values
        val bv = if (b.values === c.values) b.values.copyOf() else b.values
        val m = c.rows
        for (j in 0 until c.cols) {
            for (i in 0 until m) {
                var sum = 0.0
                for (p in 0 until depth) {
                    val left = if (transposeA) av[p + i * a.rows] else av[i + p * a.rows]
                    val right = if (transposeB) bv[j + p * b.rows] else bv[p + j * b.rows]
                    sum += left * right
                }
                val index = i + j * m
                c.values[index] = alpha * sum + scaled(beta, c.values[index])
            }
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c, symmetricStructure(lower))
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        val av = if (a.values === c.values) a.values.copyOf() else a.values
        val bv = if (b.values === c.values) b.values.copyOf() else b.values
        forTriangle(c.rows, lower) { i, j ->
            var sum = 0.0
            for (p in 0 until depth) {
                val left = if (transposeA) av[p + i * a.rows] else av[i + p * a.rows]
                val right = if (transposeB) bv[j + p * b.rows] else bv[p + j * b.rows]
                sum += left * right
            }
            val index = i + j * c.rows
            c.values[index] = alpha * sum + scaled(beta, c.values[index])
        }
    }

    private fun symmetric(values: DoubleArray, n: Int, i: Int, j: Int, lower: Boolean): Double =
        if (lower == (i >= j)) values[i + j * n] else values[j + i * n]

    override fun symv(alpha: Double, a: DenseMatrix, x: DoubleArray, beta: Double, y: DoubleArray, lower: Boolean) {
        requireSymvOperands(a, symmetricStructure(lower), x.asVector(), y.asVector())
        if (alpha == 0.0) {
            scale(y, beta)
            return
        }
        val av = if (a.values === y) a.values.copyOf() else a.values
        val xv = if (x === y) x.copyOf() else x
        for (i in y.indices) {
            var sum = 0.0
            for (j in xv.indices) sum += symmetric(av, a.rows, i, j, lower) * xv[j]
            y[i] = alpha * sum + scaled(beta, y[i])
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        right: Boolean,
    ) {
        requireSymmOperands(a, symmetricStructure(lower), b, c, right)
        if (alpha == 0.0) {
            scale(c.values, beta)
            return
        }
        val av = if (a.values === c.values) a.values.copyOf() else a.values
        val bv = if (b.values === c.values) b.values.copyOf() else b.values
        for (j in 0 until c.cols) {
            for (i in 0 until c.rows) {
                var sum = 0.0
                val depth = if (right) c.cols else c.rows
                for (p in 0 until depth) {
                    sum += if (right) {
                        bv[i + p * b.rows] * symmetric(av, a.rows, p, j, lower)
                    } else {
                        symmetric(av, a.rows, i, p, lower) * bv[p + j * b.rows]
                    }
                }
                val index = i + j * c.rows
                c.values[index] = alpha * sum + scaled(beta, c.values[index])
            }
        }
    }

    override fun syrk(alpha: Double, a: DenseMatrix, transpose: Boolean, beta: Double, c: DenseMatrix, lower: Boolean) {
        requireSyrkOperands(a, transpose, c, symmetricStructure(lower))
        productTriangle(alpha, a, a, transpose, beta, c, lower, doubled = false)
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
    ) {
        requireSyr2kOperands(a, b, transpose, c, symmetricStructure(lower))
        productTriangle(alpha, a, b, transpose, beta, c, lower, doubled = true)
    }

    private fun productTriangle(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrix,
        lower: Boolean,
        doubled: Boolean,
    ) {
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleTriangle(c, beta, lower)
            return
        }
        val av = if (a.values === c.values) a.values.copyOf() else a.values
        val bv = if (b.values === c.values) b.values.copyOf() else b.values
        forTriangle(c.rows, lower) { i, j ->
            var sum = 0.0
            for (p in 0 until depth) {
                val aip = if (transpose) av[p + i * a.rows] else av[i + p * a.rows]
                val ajp = if (transpose) av[p + j * a.rows] else av[j + p * a.rows]
                if (doubled) {
                    val bip = if (transpose) bv[p + i * b.rows] else bv[i + p * b.rows]
                    val bjp = if (transpose) bv[p + j * b.rows] else bv[j + p * b.rows]
                    sum += aip * bjp + bip * ajp
                } else {
                    sum += aip * ajp
                }
            }
            val index = i + j * c.rows
            c.values[index] = alpha * sum + scaled(beta, c.values[index])
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrix) {
        requireGerOperands(x.asVector(), y.asVector(), a)
        if (alpha == 0.0) return
        val xv = if (x === a.values) x.copyOf() else x
        val yv = if (y === a.values) y.copyOf() else y
        for (j in yv.indices) for (i in xv.indices) a.values[i + j * a.rows] += alpha * xv[i] * yv[j]
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyrOperands(a, symmetricStructure(lower), "syr", x)
        if (alpha == 0.0) return
        val xv = x.toDoubleArray()
        forTriangle(a.rows, lower) { i, j -> a.values[i + j * a.rows] += alpha * xv[i] * xv[j] }
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, lower: Boolean) {
        requireSyrOperands(a, symmetricStructure(lower), "syr2", x, y)
        if (alpha == 0.0) return
        val xv = x.toDoubleArray()
        val yv = y.toDoubleArray()
        forTriangle(a.rows, lower) { i, j ->
            a.values[i + j * a.rows] += alpha * (xv[i] * yv[j] + yv[i] * xv[j])
        }
    }

    private fun triangular(values: DoubleArray, n: Int, i: Int, j: Int, transpose: Boolean, unitDiag: Boolean): Double {
        if (unitDiag && i == j) return 1.0
        return if (transpose) values[j + i * n] else values[i + j * n]
    }

    override fun trsv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.asVector(), "trsv")
        val av = if (a.values === x) a.values.copyOf() else a.values
        solve(av, a.rows, x, lower, transpose, unitDiag)
    }

    private fun solve(a: DoubleArray, n: Int, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        val forward = lower != transpose
        val order = if (forward) 0 until n else n - 1 downTo 0
        for (i in order) {
            var sum = x[i]
            val inner = if (forward) 0 until i else i + 1 until n
            for (j in inner) sum -= triangular(a, n, i, j, transpose, unitDiag) * x[j]
            x[i] = sum / triangular(a, n, i, i, transpose, unitDiag)
        }
    }

    override fun trmv(a: DenseMatrix, x: DoubleArray, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
        requireTriangularVectorOperands(a, triangle(lower, unitDiag), x.asVector(), "trmv")
        val av = if (a.values === x) a.values.copyOf() else a.values
        val source = x.copyOf()
        for (i in x.indices) {
            var sum = 0.0
            for (j in source.indices) {
                val inTriangle = if (lower != transpose) j <= i else j >= i
                if (inTriangle) sum += triangular(av, a.rows, i, j, transpose, unitDiag) * source[j]
            }
            x[i] = sum
        }
    }

    override fun trsm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true)

    override fun trmm(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false)

    private fun triangularMatrix(
        a: DenseMatrix,
        b: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireTriangularMatrixOperands(a, triangle(lower, unitDiag), b, right, what)
        if (alpha == 0.0) {
            b.values.fill(0.0)
            return
        }
        val av = if (a.values === b.values) a.values.copyOf() else a.values
        val sides = if (right) b.rows else b.cols
        val order = a.rows
        for (s in 0 until sides) {
            val rhs = DoubleArray(
                order,
            ) { k -> alpha * if (right) b.values[s + k * b.rows] else b.values[k + s * b.rows] }
            val flipped = if (right) !transpose else transpose
            if (solve) {
                solve(av, order, rhs, lower, flipped, unitDiag)
            } else {
                multiplyTriangle(av, order, rhs, lower, flipped, unitDiag)
            }
            for (k in 0 until order) if (right) b.values[s + k * b.rows] = rhs[k] else b.values[k + s * b.rows] = rhs[k]
        }
    }

    private fun multiplyTriangle(
        a: DoubleArray,
        n: Int,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val source = x.copyOf()
        for (i in 0 until n) {
            var sum = 0.0
            for (j in 0 until n) {
                val inTriangle = if (lower != transpose) j <= i else j >= i
                if (inTriangle) sum += triangular(a, n, i, j, transpose, unitDiag) * source[j]
            }
            x[i] = sum
        }
    }

    private fun triangle(lower: Boolean, unitDiag: Boolean): MatrixStructure = when {
        unitDiag && lower -> MatrixStructure.UnitLower
        unitDiag -> MatrixStructure.UnitUpper
        lower -> MatrixStructure.TriangularLower
        else -> MatrixStructure.TriangularUpper
    }

    private fun scale(values: DoubleArray, beta: Double) {
        if (beta == 0.0) values.fill(0.0) else for (i in values.indices) values[i] *= beta
    }

    private fun scaleTriangle(c: DenseMatrix, beta: Double, lower: Boolean) = forTriangle(c.rows, lower) { i, j ->
        val index = i + j * c.rows
        c.values[index] = scaled(beta, c.values[index])
    }

    private inline fun forTriangle(n: Int, lower: Boolean, action: (Int, Int) -> Unit) {
        for (j in 0 until n) for (i in 0 until n) if ((lower && i >= j) || (!lower && i <= j)) action(i, j)
    }
}
