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
    "copy-overlap" to OperationSpec(1),
    "axpy-views" to OperationSpec(1),
    "spdot-view" to OperationSpec(1, fixture = "sparse-uniform", required = setOf("density")),
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
    "spdot" to OperationSpec(1, IndexedDot, fixture = "sparse-uniform", required = setOf("density")),
    "spdot-raw" to OperationSpec(1, IndexedDot, fixture = "sparse-uniform", required = setOf("density")),
    "spdot-sparse" to OperationSpec(1, IndexedSparseDot, fixture = "sparse-uniform", required = setOf("density")),
    "spaxpy" to OperationSpec(1, IndexedAxpy, fixture = "sparse-uniform", required = setOf("density")),
    "spaxpy-raw" to OperationSpec(1, IndexedAxpy, fixture = "sparse-uniform", required = setOf("density")),
    "spdot-column" to OperationSpec(1, fixture = "sparse-uniform", required = setOf("density")),
    "spdot-column-raw" to OperationSpec(1, fixture = "sparse-uniform", required = setOf("density")),
    "spnrm2" to OperationSpec(1, Nrm2, fixture = "sparse-uniform", required = setOf("density")),
    "spasum" to OperationSpec(1, Asum, fixture = "sparse-uniform", required = setOf("density")),
    "spscatter" to OperationSpec(1, IndexedScatter, fixture = "sparse-uniform", required = setOf("density")),
    "spscatter-raw" to OperationSpec(1, IndexedScatter, fixture = "sparse-uniform", required = setOf("density")),
    "spgather" to OperationSpec(
        1,
        IndexedGather,
        fixture = "sparse-uniform",
        required = setOf("density"),
        optional = setOf("timing"),
    ),
    "spgather-zero" to OperationSpec(1, IndexedGatherZero, fixture = "sparse-uniform", required = setOf("density")),
    "spdot-slice" to OperationSpec(1, IndexedDot, fixture = "sparse-uniform", required = setOf("density")),
    "spaxpy-slice" to OperationSpec(1, IndexedAxpy, fixture = "sparse-uniform", required = setOf("density")),
    "spscatter-slice" to OperationSpec(1, IndexedScatter, fixture = "sparse-uniform", required = setOf("density")),
    "spbuild-triplets" to OperationSpec(2, fixture = "sparse-uniform", required = setOf("density"), optional = setOf("support")),
    "spbuild-duplicates" to OperationSpec(
        2,
        fixture = "sparse-uniform",
        required = setOf("density"),
        optional = setOf("support"),
    ),
    "spgemv" to OperationSpec(
        2,
        fixture = "sparse-uniform",
        required = setOf("density", "mode"),
        optional = setOf("support", "transA"),
    ),
    "spmm" to OperationSpec(
        3,
        fixture = "sparse-uniform",
        required = setOf("density", "mode"),
        optional = setOf("support", "transA", "transB"),
    ),
    "spmm-right" to OperationSpec(
        3,
        fixture = "sparse-uniform",
        required = setOf("density", "mode"),
        optional = setOf("support", "transA"),
    ),
    "spsymv" to OperationSpec(
        1,
        fixture = "sparse-triangular",
        required = setOf("density", "uplo", "mode"),
        optional = setOf("support"),
    ),
    "spsymm" to OperationSpec(
        2,
        fixture = "sparse-triangular",
        required = setOf("density", "side", "uplo", "mode"),
        optional = setOf("support"),
    ),
    "sptrsv" to OperationSpec(
        1,
        fixture = "sparse-triangular",
        required = setOf("density", "uplo", "transA", "diag", "mode"),
        optional = setOf("support"),
    ),
    "sptrmv" to OperationSpec(
        1,
        fixture = "sparse-triangular",
        required = setOf("density", "uplo", "transA", "diag", "mode"),
        optional = setOf("support"),
    ),
    "sptrsm" to OperationSpec(
        2,
        fixture = "sparse-triangular",
        required = setOf("density", "side", "uplo", "transA", "diag", "mode"),
        optional = setOf("support"),
    ),
    "sptrmm" to OperationSpec(
        2,
        fixture = "sparse-triangular",
        required = setOf("density", "side", "uplo", "transA", "diag", "mode"),
        optional = setOf("support"),
    ),
    "spsyrk-dense" to OperationSpec(
        2,
        fixture = "sparse-uniform",
        required = setOf("density", "uplo"),
        optional = setOf("support"),
    ),
)

internal val BenchCase.spec: OperationSpec get() = operationSpecs.getValue(operation)

/** Transpose changes attribution only where the kernel seam distinguishes it. */
internal val BenchCase.implementation: EngineImplementation?
    get() = if (flag("transA")) spec.transposed ?: spec.implementation else spec.implementation
