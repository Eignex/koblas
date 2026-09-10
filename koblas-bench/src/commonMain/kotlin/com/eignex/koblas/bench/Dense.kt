@file:OptIn(com.eignex.koblas.ExperimentalKoblasApi::class)

package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.Workspace
import com.eignex.koblas.compensatedSum
import com.eignex.koblas.iamax
import com.eignex.koblas.dense.PackedPanels

internal class CaseWork(
    val comparisonKind: String,
    val timingMode: String,
    val run: () -> Double,
    val close: () -> Unit = {},
)

internal fun denseWork(case: BenchCase, engine: KoblasContext): CaseWork? {
    val d = case.dimensions
    val vectors = engine.vectorKernels
    val panels = engine.panelKernels
    val alpha = 0.875
    val beta = -0.25
    return when (case.operation) {
        "dot" -> {
            val x = Fixtures.vector(d[0], 1); val y = Fixtures.vector(d[0], 2)
            CaseWork("direct", "arithmetic", { vectors.dot(x, 0, y, 0, d[0]) })
        }
        "axpy" -> {
            val x = Fixtures.vector(d[0], 1); val initial = Fixtures.vector(d[0], 2); val y = initial.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { initial.copyInto(y); vectors.axpy(y, 0, alpha, x, 0, d[0]); y[0] })
        }
        "scal" -> {
            val initial = Fixtures.vector(d[0], 1); val x = initial.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { initial.copyInto(x); vectors.scale(x, 0, alpha, d[0]); x[0] })
        }
        "nrm2" -> vectorReduction(d[0], "direct") { x -> vectors.nrm2(x, 0, x.size) }
        "asum" -> vectorReduction(d[0], "direct") { x -> vectors.asum(x, 0, x.size) }
        "sum" -> vectorReduction(d[0], "direct") { x -> vectors.sum(x, 0, x.size) }
        "compensated-sum" -> vectorReduction(d[0], "unsupported") { x -> DenseVector.wrap(x).compensatedSum() }
        "iamax" -> vectorReduction(d[0], "direct") { x -> DenseVector.wrap(x).iamax().toDouble() }
        "ssqd" -> {
            val x = Fixtures.vector(d[0], 1); val y = Fixtures.vector(d[0], 2)
            CaseWork("unsupported", "arithmetic", { vectors.ssqd(x, 0, y, 0, d[0]) })
        }
        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(d[0], 2); val x = x0.copyOf(); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { x0.copyInto(x); y0.copyInto(y); vectors.swap(x, 0, y, 0, d[0]); x[0] + y[0] })
        }
        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(d[0], 2); val x = x0.copyOf(); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { x0.copyInto(x); y0.copyInto(y); vectors.rot(x, 0, y, 0, d[0], 0.8, 0.6); x[0] + y[0] })
        }
        "rotm" -> {
            val x0 = Fixtures.vector(d[0], 1); val y0 = Fixtures.vector(d[0], 2); val x = x0.copyOf(); val y = y0.copyOf()
            val transform = vectors.rotmg(1.0, 1.0, 2.0, 1.0)
            CaseWork("direct", "reset-and-arithmetic", { x0.copyInto(x); y0.copyInto(y); vectors.rotm(x, 0, 1, y, 0, 1, d[0], transform); x[0] + y[0] })
        }
        "rotmg" -> CaseWork("direct", "arithmetic", { vectors.rotmg(1.0, 1.0, 2.0, 1.0).let { it.d1 + it.d2 + it.x1 + it.flag } })
        "dot4" -> {
            val x = Fixtures.vector(d[0], 1); val a = Fixtures.vector(d[0] * 4, 2); val out = DoubleArray(4)
            CaseWork("composed", "arithmetic", { panels.dot4(a, 0, d[0], x, 0, d[0], out, 0); out.sum() })
        }
        "axpy4" -> {
            val x = Fixtures.vector(d[0] * 4, 1); val y0 = Fixtures.vector(d[0], 2); val y = y0.copyOf()
            CaseWork("composed", "reset-and-arithmetic", {
                y0.copyInto(y); panels.axpy4(y, 0, x, 0, d[0], alpha, -alpha, alpha, -alpha, d[0]); y[0]
            })
        }
        "dot-axpy" -> {
            val x = Fixtures.vector(d[0], 1); val z = Fixtures.vector(d[0], 2); val y0 = Fixtures.vector(d[0], 3); val y = y0.copyOf()
            CaseWork("composed", "reset-and-arithmetic", { y0.copyInto(y); panels.dotAxpy(y, 0, alpha, x, 0, z, 0, d[0]) + y[0] })
        }
        "gemv" -> {
            val m = d[0]; val n = d[1]; val trans = case.flag("transA")
            val a = Fixtures.matrix(if (trans) n else m, if (trans) m else n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(m, 3); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { y0.copyInto(y); engine.gemv(alpha, a, x, beta, y, trans); y[0] })
        }
        "symv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val a = Fixtures.matrix(n, n, 1)
            val x = Fixtures.vector(n, 2); val y0 = Fixtures.vector(n, 3); val y = y0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", { y0.copyInto(y); engine.symv(alpha, a, x, beta, y, lower); y[0] })
        }
        "ger" -> matrixUpdate(d[0], d[1], "direct") { matrix, x, y -> engine.ger(alpha, x, y, matrix) }
        "syr" -> symmetricUpdate(d[0], case.option("uplo", "L") == "L", "direct") { matrix, x, _ -> engine.syr(alpha, DenseVector.wrap(x), matrix, case.option("uplo", "L") == "L") }
        "syr2" -> symmetricUpdate(d[0], case.option("uplo", "L") == "L", "direct") { matrix, x, y -> engine.syr2(alpha, DenseVector.wrap(x), DenseVector.wrap(y), matrix, case.option("uplo", "L") == "L") }
        "trsv", "trmv" -> {
            val n = d[0]; val lower = case.option("uplo", "L") == "L"; val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"
            val a = Fixtures.triangular(n, 1, lower); val x0 = Fixtures.vector(n, 2); val x = x0.copyOf()
            CaseWork("direct", "reset-and-arithmetic", {
                x0.copyInto(x)
                if (case.operation == "trsv") engine.trsv(a, x, lower, trans, unit) else engine.trmv(a, x, lower, trans, unit)
                x[0]
            })
        }
        "gemm" -> gemmWork(case, engine)
        "symm" -> {
            val m = d[0]; val n = d[1]; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (right) n else m, if (right) n else m, 1); val b = Fixtures.matrix(m, n, 2)
            val c0 = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3); val workspace = Workspace()
            CaseWork("direct", "reset-and-arithmetic", { c0.data.copyInto(c.data); engine.symm(alpha, a, b, beta, c, lower, right, workspace); c.data[0] })
        }
        "gemmt" -> {
            val n = d[0]; val k = d[1]; val ta = case.flag("transA"); val tb = case.flag("transB"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (ta) k else n, if (ta) n else k, 1); val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3); val workspace = Workspace()
            CaseWork("direct", "reset-and-arithmetic", { c0.data.copyInto(c.data); engine.gemmt(alpha, a, ta, b, tb, beta, c, lower, workspace); c.data[0] })
        }
        "syrk", "syr2k" -> {
            val n = d[0]; val k = d[1]; val trans = case.flag("transA"); val lower = case.option("uplo", "L") == "L"
            val a = Fixtures.matrix(if (trans) k else n, if (trans) n else k, 1); val b = Fixtures.matrix(a.rows, a.cols, 2)
            val c0 = Fixtures.matrix(n, n, 3); val c = Fixtures.matrix(n, n, 3); val workspace = Workspace()
            CaseWork("direct", "reset-and-arithmetic", {
                c0.data.copyInto(c.data)
                if (case.operation == "syrk") engine.syrk(alpha, a, trans, beta, c, lower, workspace)
                else engine.syr2k(alpha, a, b, trans, beta, c, lower, workspace)
                c.data[0]
            })
        }
        "trsm", "trmm" -> triangularMatrixWork(case, engine)
        else -> packedWork(case, engine)
    }
}

private fun vectorReduction(size: Int, comparison: String, operation: (DoubleArray) -> Double): CaseWork {
    val x = Fixtures.vector(size, 1)
    return CaseWork(comparison, "arithmetic", { operation(x) })
}

private fun matrixUpdate(rows: Int, cols: Int, comparison: String, operation: (DenseMatrix, DoubleArray, DoubleArray) -> Unit): CaseWork {
    val original = Fixtures.matrix(rows, cols, 1); val target = Fixtures.matrix(rows, cols, 1)
    val x = Fixtures.vector(rows, 2); val y = Fixtures.vector(cols, 3)
    return CaseWork(comparison, "reset-and-arithmetic", { original.data.copyInto(target.data); operation(target, x, y); target.data[0] })
}

private fun symmetricUpdate(size: Int, lower: Boolean, comparison: String, operation: (DenseMatrix, DoubleArray, DoubleArray) -> Unit): CaseWork {
    val original = Fixtures.matrix(size, size, 1); val target = Fixtures.matrix(size, size, 1)
    val x = Fixtures.vector(size, 2); val y = Fixtures.vector(size, 3)
    return CaseWork(comparison, "reset-and-arithmetic", { original.data.copyInto(target.data); operation(target, x, y); target.data[if (lower) 0 else size * size - 1] })
}

private fun gemmWork(case: BenchCase, engine: KoblasContext): CaseWork {
    val (m, n, k) = case.dimensions; val ta = case.flag("transA"); val tb = case.flag("transB")
    val a = Fixtures.matrix(if (ta) k else m, if (ta) m else k, 1)
    val b = Fixtures.matrix(if (tb) n else k, if (tb) k else n, 2)
    val original = Fixtures.matrix(m, n, 3); val c = Fixtures.matrix(m, n, 3); val workspace = Workspace()
    return CaseWork("direct", "reset-and-arithmetic", {
        original.data.copyInto(c.data); engine.gemm(0.875, a, ta, b, tb, -0.25, c, workspace); c.data[0]
    })
}

private fun triangularMatrixWork(case: BenchCase, engine: KoblasContext): CaseWork {
    val (m, n) = case.dimensions; val right = case.option("side", "L") == "R"; val lower = case.option("uplo", "L") == "L"
    val trans = case.flag("transA"); val unit = case.option("diag", "N") == "U"; val order = if (right) n else m
    val triangle = Fixtures.triangular(order, 1, lower); val original = Fixtures.matrix(m, n, 2); val b = Fixtures.matrix(m, n, 2); val workspace = Workspace()
    return CaseWork("direct", "reset-and-arithmetic", {
        original.data.copyInto(b.data)
        if (case.operation == "trsm") engine.trsm(triangle, b, lower, trans, unit, right, 0.875, workspace)
        else engine.trmm(triangle, b, lower, trans, unit, right, 0.875, workspace)
        b.data[0]
    })
}

private fun packedWork(case: BenchCase, engine: KoblasContext): CaseWork? {
    if (case.operation !in setOf(
            "gemm-tile", "packed-trsm", "gemm-trsm", "pack-left", "pack-right", "pack-symmetric-left",
            "pack-symmetric-right", "pack-triangular-left", "pack-triangular-right", "write-left", "write-right",
            "clear-left-padding", "clear-right-padding",
        )) return null
    val packed = engine.packedKernels
    val panels = engine.packedPanels
    val physical = case.option("physical", "4x4").split('x').map(String::toInt)
    val layoutOperation = case.operation.startsWith("pack-") || case.operation.startsWith("write-") || case.operation.startsWith("clear-")
    val actualRows = if (layoutOperation) panels.tileRows else packed.gemmTileRows
    val actualColumns = if (layoutOperation) panels.tileColumns else packed.gemmTileCols
    if (actualRows != physical[0] || actualColumns != physical[1]) return null
    val rows = case.dimension(0); val second = case.dimension(1); val depth = if (case.dimensions.size == 3) case.dimension(2) else second
    val leftDepth = if (layoutOperation) second else depth
    val rightDepth = if (layoutOperation) rows else depth
    val left = DoubleArray(leftDepth * physical[0]).also { Fixtures.vector(it.size, 1).copyInto(it) }
    val right = DoubleArray(rightDepth * physical[1]).also { Fixtures.vector(it.size, 2).copyInto(it) }
    if (!layoutOperation && case.operation != "packed-trsm") {
        for (p in 0 until depth) for (i in rows until physical[0]) left[i + p * physical[0]] = 0.0
        for (p in 0 until depth) for (j in second until physical[1]) right[j + p * physical[1]] = 0.0
    }
    val triangle = DoubleArray(physical[1] * physical[1])
    val lower = case.option("uplo", "L") == "L"; val unit = case.option("diag", "N") == "U"
    if (case.operation == "packed-trsm" || case.operation == "gemm-trsm") {
        packedTriangleFixture(second, physical[1], lower).copyInto(triangle)
    }
    val output0 = Fixtures.vector(physical[0] * physical[1], 4); val output = output0.copyOf()
    val timing = if (layoutOperation) "layout" else "arithmetic-only"
    val comparison = when (case.operation) { "gemm-tile", "packed-trsm" -> "partial"; "gemm-trsm" -> "composed"; else -> "unsupported" }
    return when (case.operation) {
        "gemm-tile" -> CaseWork(comparison, timing, { output0.copyInto(output); packed.gemmTile(depth, left, 0, right, 0, output, 0, physical[0]); output[0] })
        "packed-trsm" -> CaseWork(comparison, timing, { output0.copyInto(output); packed.trsmTile(rows, second, triangle, 0, lower, unit, output, 0); output[0] })
        "gemm-trsm" -> CaseWork(comparison, timing, { output0.copyInto(output); packed.gemmTrsmTile(depth, rows, second, left, 0, right, 0, triangle, 0, lower, unit, output, 0); output[0] })
        else -> packedLayoutWork(case, panels, rows, second, left, right, lower, unit)
    }
}

internal fun packedTriangleFixture(order: Int, physicalColumns: Int, lower: Boolean): DoubleArray {
    val logical = Fixtures.triangular(order, 20, lower)
    return DoubleArray(physicalColumns * physicalColumns).also { packed ->
        for (i in 0 until order) for (j in 0 until order) packed[i * physicalColumns + j] = logical[i, j]
    }
}

private fun packedLayoutWork(
    case: BenchCase, panels: PackedPanels, first: Int, second: Int,
    left: DoubleArray, right: DoubleArray, lower: Boolean, unit: Boolean,
): CaseWork {
    val isLeft = "left" in case.operation
    val sourceRows = if (isLeft) first else first
    val sourceCols = if (isLeft) second else second
    val source = if (case.operation.contains("triangular")) Fixtures.triangular(first, 1, lower) else Fixtures.matrix(sourceRows, sourceCols, 1)
    val destination = DenseMatrix.zero(sourceRows, sourceCols)
    val panel = if (isLeft) left else right
    return CaseWork("unsupported", "layout", {
        when (case.operation) {
            "pack-left" -> panels.packLeft(source, panel, first, second)
            "pack-right" -> panels.packRight(source, panel, first, second)
            "pack-symmetric-left" -> panels.packSymmetricLeft(source, panel, first, second, lower)
            "pack-symmetric-right" -> panels.packSymmetricRight(source, panel, first, second, lower)
            "pack-triangular-left" -> panels.packTriangularLeft(source, panel, first, second, lower, unitDiagonal = unit)
            "pack-triangular-right" -> panels.packTriangularRight(source, panel, first, second, lower, unitDiagonal = unit)
            "write-left" -> panels.writeLeft(panel, destination, first, second)
            "write-right" -> panels.writeRight(panel, destination, first, second)
            "clear-left-padding" -> panels.clearLeftPadding(panel, first, second)
            "clear-right-padding" -> panels.clearRightPadding(panel, first, second)
        }
        if (case.operation.startsWith("write")) destination.data[0] else panel[0]
    })
}
