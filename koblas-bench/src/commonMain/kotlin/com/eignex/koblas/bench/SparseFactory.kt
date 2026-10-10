package com.eignex.koblas.bench

import com.eignex.koblas.SparseMatrix

/** Portable triplet construction, including ordering, duplicate reduction and ownership of the result. */
internal fun sparseFactoryWork(case: BenchCase): CaseWork {
    val rows = case.dimension(0)
    val columns = case.dimension(1)
    val fixture = Fixtures.sparse(rows, columns, case.option("density", "0.01").toDouble(), 1,
        support = case.option("support", "uniform"))
    val sourceRows = fixture.copyRowIndices()
    val pointers = fixture.copyColumnPointers()
    val sourceColumns = IntArray(fixture.nnz)
    for (j in 0 until columns) for (k in pointers[j] until pointers[j + 1]) sourceColumns[k] = j
    val copies = if (case.operation == "spbuild-duplicates") 2 else 1
    val indices = IntArray(copies * fixture.nnz)
    val columnIndices = IntArray(indices.size)
    val values = DoubleArray(indices.size)
    // Reverse the coordinates so the timed call must order them, including the duplicate runs.
    for (k in indices.indices) {
        val source = fixture.nnz - 1 - k / copies
        indices[k] = sourceRows[source]
        columnIndices[k] = sourceColumns[source]
        values[k] = fixture.values[source] / copies
    }
    fun build() = SparseMatrix.ofTriplets(rows, columns, indices, columnIndices, values)
    val expected = Array(rows) { DoubleArray(columns) }
    val support = Array(rows) { BooleanArray(columns) }
    for (k in indices.indices) {
        val i = indices[k]
        val j = columnIndices[k]
        expected[i][j] = if (support[i][j]) expected[i][j] + values[k] else values[k]
        support[i][j] = true
    }
    var result = build()
    SparseReference.checkSparse(expected, support, result, case.id)
    return CaseWork("composed", "construction", {
        // Keep the whole result observable, including structural arrays a numerical sink would not read.
        result = build()
        result.values.firstOrNull() ?: result.nnz.toDouble()
    }, kernel = "portable-csc/ofTriplets")
}
