package com.eignex.koblas.internal.storage

import com.eignex.koblas.Matrix
import com.eignex.koblas.Vector

/** Opaque matrix storage exposes its column through the same indexed mutable contract. */
internal class MatrixColumnVector(private val matrix: Matrix, private val column: Int) : Vector {
    override val size: Int get() = matrix.rows

    override fun get(i: Int): Double = matrix[i, column]

    override fun set(i: Int, v: Double) {
        matrix[i, column] = v
    }
}
