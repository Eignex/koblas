package com.eignex.koblas.bench

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.EngineImplementation.*

/** Case syntax and arithmetic identity share one specification; shapes remain routine-specific. */
internal class OperationSpec(
    val dimensions: Int,
    val implementation: EngineImplementation? = null,
    val fixture: String = "uniform",
    val required: Set<String> = emptySet(),
    optional: Set<String> = emptySet(),
    val minimumOrder: Int = 1,
    val level: Int = if (implementation?.vector == true) 1 else 2,
    val transposed: EngineImplementation? = null,
) {
    val allowed: Set<String> = required + optional
}

internal val operationSpecs: Map<String, OperationSpec> = mapOf(
    "dot" to OperationSpec(1, Dot, optional = setOf("stride")),
    "axpy" to OperationSpec(1, Axpy, optional = setOf("stride", "timing")),
    "axpy-unit" to OperationSpec(1, Axpy),
    "scal" to OperationSpec(1, Scale, optional = setOf("stride", "timing")),
    "nrm2" to OperationSpec(1, Nrm2, optional = setOf("stride")),
    "asum" to OperationSpec(1, Asum, optional = setOf("stride")),
    "sum" to OperationSpec(1, Sum, optional = setOf("stride")),
    "iamax" to OperationSpec(1, Iamax, optional = setOf("stride")),
    "swap" to OperationSpec(1, Swap, optional = setOf("stride", "timing")),
    "rot" to OperationSpec(1, Rot, optional = setOf("stride", "timing")),
    "copy" to OperationSpec(1, Copy, optional = setOf("stride")),
    "gemv" to OperationSpec(2, Gemv, optional = setOf("transA", "timing"), transposed = GemvTransposed),
    "symv" to OperationSpec(1, Symv, required = setOf("uplo"), optional = setOf("timing")),
    "ger" to OperationSpec(2, Ger, optional = setOf("timing")),
    "syr" to OperationSpec(1, Syr, required = setOf("uplo"), optional = setOf("timing")),
    "syr2" to OperationSpec(1, Syr2, required = setOf("uplo"), optional = setOf("timing")),
    "trsv" to OperationSpec(
        1,
        Trsv,
        fixture = "triangular",
        required = setOf("uplo", "transA", "diag"),
        transposed = TrsvTransposed,
    ),
    "ger-shrinking" to OperationSpec(1, minimumOrder = 2),
    "ger-window-shrinking" to OperationSpec(1, minimumOrder = 2),
    "trmv" to OperationSpec(
        1,
        Trmv,
        fixture = "triangular",
        required = setOf("uplo", "transA", "diag"),
        transposed = TrmvTransposed,
    ),
    "gemm" to OperationSpec(3, Gemm, optional = setOf("transA", "transB", "workspace", "layout"), level = 3),
    "symm" to OperationSpec(2, Symm, required = setOf("side", "uplo"), level = 3),
    "gemmt" to OperationSpec(2, Gemmt, required = setOf("uplo", "transA", "transB"), level = 3),
    "syrk" to OperationSpec(2, Syrk, required = setOf("uplo", "transA"), level = 3),
    "syr2k" to OperationSpec(2, Syr2k, required = setOf("uplo", "transA"), level = 3),
    "trsm" to OperationSpec(2, Trsm, fixture = "triangular", required = setOf("side", "uplo", "transA", "diag"), level = 3),
    "trmm" to OperationSpec(2, Trmm, fixture = "triangular", required = setOf("side", "uplo", "transA", "diag"), level = 3),
    "spdot" to sparseVector(IndexedDot),
    "spdot-raw" to sparseVector(IndexedDot),
    "spdot-sparse" to sparseVector(IndexedSparseDot),
    "spaxpy" to sparseVector(IndexedAxpy),
    "spaxpy-raw" to sparseVector(IndexedAxpy),
    "spdot-column" to sparseVector(),
    "spdot-column-raw" to sparseVector(),
    "spnrm2" to sparseVector(Nrm2),
    "spasum" to sparseVector(Asum),
    "spscatter" to sparseVector(IndexedScatter),
    "spscatter-raw" to sparseVector(IndexedScatter),
    "spgather" to sparseVector(IndexedGather, setOf("timing")),
    "spgather-zero" to sparseVector(IndexedGatherZero),
    "spdot-slice" to sparseVector(IndexedDot),
    "spaxpy-slice" to sparseVector(IndexedAxpy),
    "spscatter-slice" to sparseVector(IndexedScatter),
    "spgemv" to sparseMatrix(2, optional = setOf("transA")),
    "spmm" to sparseMatrix(3, optional = setOf("transA", "transB")),
    "spmm-right" to sparseMatrix(3, optional = setOf("transA")),
    "spsymv" to sparseMatrix(1, setOf("mode", "uplo"), triangular = true),
    "spsymm" to sparseMatrix(2, setOf("mode", "side", "uplo"), triangular = true),
    "sptrsv" to sparseMatrix(1, setOf("mode", "uplo", "transA", "diag"), triangular = true),
    "sptrmv" to sparseMatrix(1, setOf("mode", "uplo", "transA", "diag"), triangular = true),
    "sptrsm" to sparseMatrix(2, setOf("mode", "side", "uplo", "transA", "diag"), triangular = true),
    "sptrmm" to sparseMatrix(2, setOf("mode", "side", "uplo", "transA", "diag"), triangular = true),
    "spsyrk-dense" to sparseMatrix(2, setOf("uplo")),
)

private fun sparseVector(implementation: EngineImplementation? = null, optional: Set<String> = emptySet()) =
    OperationSpec(1, implementation, "sparse-uniform", setOf("density"), optional)

private fun sparseMatrix(
    dimensions: Int,
    required: Set<String> = setOf("mode"),
    optional: Set<String> = emptySet(),
    triangular: Boolean = false,
) = OperationSpec(
    dimensions, fixture = if (triangular) "sparse-triangular" else "sparse-uniform",
    required = setOf("density") + required, optional = setOf("support") + optional,
)

internal val BenchCase.spec: OperationSpec get() = operationSpecs.getValue(operation)

/** Transpose changes attribution only where the kernel seam distinguishes it. */
internal val BenchCase.implementation: EngineImplementation?
    get() = if (flag("transA")) spec.transposed ?: spec.implementation else spec.implementation
