package com.eignex.koblas

internal fun requireHfactorShape(condition: Boolean, message: () -> String) {
    if (!condition) throw DimensionMismatch(message())
}

internal fun requireHfactorIndex(condition: Boolean, message: () -> String) {
    if (!condition) throw IndexOutOfBoundsException(message())
}

internal fun requireHfactorSolveShapes(rows: Int, cols: Int, b: DoubleArray, out: DoubleArray) {
    requireHfactorShape(b.size == rows) { "solve: b size ${b.size}, expected $rows" }
    requireHfactorShape(out.size == cols) { "solve: out size ${out.size}, expected $cols" }
}

internal fun requireHfactorSolveShapes(rows: Int, cols: Int, b: DenseMatrix, out: DenseMatrix) {
    requireHfactorShape(b.rows == rows) { "solve: B has ${b.rows} rows, expected $rows" }
    requireHfactorShape(out.rows == cols && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${cols}x${b.cols}"
    }
}
