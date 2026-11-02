package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines

import com.eignex.koblas.EngineImplementation

import com.eignex.koblas.KoblasEngine

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.vendor.Vendor

/**
 * The work for one benchmark case on one arm, or the reason there is none to time.
 *
 * Rejected arms leave the row untimed so measurements remain attributable to the requested implementation.
 */
internal class ArmChoice(val work: CaseWork?, val reason: String?)

/** The prefix a vendor mode carries for each runtime, which is also which entry point may run it. */
internal const val JVM_VENDOR_PREFIX: String = "jvm-vendor-"

/** The Native runtime's vendor-mode prefix. */
internal const val NATIVE_VENDOR_PREFIX: String = "native-vendor-"

/** The vendor a benchmark mode names, or null when the mode is not a vendor arm. */
internal fun vendorFromMode(mode: String): Vendor? {
    val name = when {
        mode.startsWith(JVM_VENDOR_PREFIX) -> mode.removePrefix(JVM_VENDOR_PREFIX)
        mode.startsWith(NATIVE_VENDOR_PREFIX) -> mode.removePrefix(NATIVE_VENDOR_PREFIX)
        else -> return null
    }
    return Vendor.entries.singleOrNull { it.name.lowercase() == name }
        ?: error("unknown vendor in mode $mode; expected one of ${Vendor.entries.map { it.name.lowercase() }}")
}

/**
 * The vendor a mode names for [prefix]'s runtime, or null when the mode is not that runtime's vendor arm.
 *
 * The runtime prefix prevents a runner from publishing another runtime's vendor mode.
 */
internal fun vendorForRuntime(mode: String, prefix: String): Vendor? =
    if (mode.startsWith(prefix)) vendorFromMode(mode) else null

/**
 * Opens the vendor a mode names, failing loudly rather than quietly measuring something else.
 *
 * The description records the resolved file, version and single-thread evidence. Libraries that cannot
 * hold to one thread are rejected; those without a thread-count entry point cannot claim confirmation.
 */
internal fun openVendorForMode(mode: String): Pair<KoblasEngine, String> {
    val vendor = vendorFromMode(mode) ?: error("$mode is not a vendor mode")
    val blas = BuiltinEngines.vendor(vendor)
        ?: error("requested vendor ${vendor.vendorName} is not installed; it cannot be benchmarked on this host")
    val description = "$mode/${blas.vendor!!.vendorName}/${blas.libraryPath}/" +
        "${blas.version}/threads=${blas.threadEvidence.label}"
    return blas to (description + blas.capabilities())
}

/**
 * The work for [case] through [blas], or a reason it is not an admissible measurement.
 *
 * [exactArmRejection] requires a direct vendor call over these operands. Cases outside the bound CBLAS
 * surface are declined, including `sum`, `ssqd`, fused panels, packed tiles and sparse operations.
 * Fixtures, scalars and timing boundaries match the other arms to keep comparisons equivalent.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun vendorArm(case: BenchCase, blas: KoblasEngine): ArmChoice {
    if (case.options["layout"] == "borrowed") {
        return ArmChoice(null, "borrowed matrix cases measure engine scheduling and eligible host composition")
    }
    val d = case.dimensions
    val alpha = 0.875
    val beta = -0.25
    val lower = case.option("uplo", "L") == "L"
    val transA = case.flag("transA")
    val transB = case.flag("transB")
    val unit = case.option("diag", "N") == "U"
    val right = case.option("side", "L") == "R"

    fun symmetric(order: Int, operand: Int): DenseMatrix = Fixtures.matrix(order, order, operand)
    fun triangle(order: Int, operand: Int): DenseMatrix = Fixtures.triangular(order, operand, lower)

    fun arm(
        operation: EngineImplementation,
        matrices: List<DenseMatrix>,
        vectors: List<DenseVector>,
        timing: String,
        run: () -> Double,
    ): ArmChoice {
        // Include both operand lists so no-work calls are rejected before timing.
        exactArmRejection(blas, operation, matrices, vectors, alpha)?.let { return ArmChoice(null, it) }
        return ArmChoice(CaseWork(
            "direct",
            timing,
            run,
            kernel = "${blas.vendor?.vendorName}/${operation.entryPoint}",
        ), null)
    }

    fun declined(what: String) = ArmChoice(null, "$what has no vendor entry point in the bound CBLAS surface")

    return when (case.operation) {
        "dot" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            val y = DenseVector.wrap(Fixtures.vector(d[0], 2))
            arm(EngineImplementation.Dot, emptyList(), listOf(x, y), "arithmetic") { blas.dot(x, y) }
        }

        "axpy" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            val initial = Fixtures.vector(d[0], 2)
            val target = initial.copyOf()
            val y = DenseVector.wrap(target)
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Alternating the sign returns y close to where it started, so no reset joins the timed work.
                var signed = alpha
                arm(EngineImplementation.Axpy, emptyList(), listOf(x, y), "arithmetic") {
                    signed = -signed; blas.axpy(signed, x, y); target[0]
                }
            } else {
                arm(EngineImplementation.Axpy, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                    initial.copyInto(target); blas.axpy(alpha, x, y); target[0]
                }
            }
        }
        "axpy-unit" -> {
            val fixture = UnitAxpy(d[0])
            val x = DenseVector.wrap(fixture.x)
            val y = DenseVector.wrap(fixture.y)
            val run = { blas.axpy(1.0, x, y); fixture.y[0] }
            fixture.verify(case.id, run)
            arm(EngineImplementation.Axpy, emptyList(), listOf(x, y), "arithmetic", run)
        }

        "scal" -> {
            val initial = Fixtures.vector(d[0], 1)
            val values = initial.copyOf()
            val x = DenseVector.wrap(values)
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Negation keeps the magnitudes normal over arbitrarily many timed invocations.
                arm(EngineImplementation.Scale, emptyList(), listOf(x), "arithmetic") {
                    blas.scale(-1.0, x); values[0] + values[values.size - 1]
                }
            } else {
                arm(EngineImplementation.Scale, emptyList(), listOf(x), "reset-and-arithmetic") {
                    initial.copyInto(values); blas.scale(alpha, x); values[0] + values[values.size - 1]
                }
            }
        }

        "nrm2" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(EngineImplementation.Nrm2, emptyList(), listOf(x), "arithmetic") { blas.nrm2(x) }
        }

        "asum" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(EngineImplementation.Asum, emptyList(), listOf(x), "arithmetic") { blas.asum(x) }
        }

        "iamax" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(EngineImplementation.Iamax, emptyList(), listOf(x), "arithmetic") { blas.iamax(x).toDouble() }
        }

        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = DenseVector.wrap(xs)
            val y = DenseVector.wrap(ys)
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // A swap is its own inverse, so the operands only ever trade places.
                arm(EngineImplementation.Swap, emptyList(), listOf(x, y), "arithmetic") { blas.swap(
                    x,
                    y,
                ); xs[0] + ys[0] }
            } else {
                arm(EngineImplementation.Swap, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                    x0.copyInto(xs); y0.copyInto(ys); blas.swap(x, y); xs[0] + ys[0]
                }
            }
        }

        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = DenseVector.wrap(xs)
            val y = DenseVector.wrap(ys)
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // A rotation preserves each pair's norm, so repeating it keeps the magnitudes normal.
                arm(EngineImplementation.Rot, emptyList(), listOf(x, y), "arithmetic") { blas.rot(x, y, 0.8,
                    0.6); xs[0] + ys[0] }
            } else {
                arm(EngineImplementation.Rot, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                    x0.copyInto(xs); y0.copyInto(ys); blas.rot(x, y, 0.8, 0.6); xs[0] + ys[0]
                }
            }
        }

        "gemv" -> {
            val m = d[0]
            val n = d[1]
            val source = Fixtures.matrix(if (transA) n else m, if (transA) m else n, 1)
            val a = source
            val x = DenseVector.wrap(Fixtures.vector(n, 2))
            val y0 = Fixtures.vector(m, 3)
            val target = y0.copyOf()
            val y = DenseVector.wrap(target)
            val resets = resets(case)
            arm(EngineImplementation.Gemv, listOf(a), emptyList(), timing(case)) {
                if (resets) y0.copyInto(target); blas.gemv(alpha, a, x, beta, y, transA); target[0]
            }
        }

        "symv" -> {
            val n = d[0]
            val a = symmetric(n, 1)
            val x = DenseVector.wrap(Fixtures.vector(n, 2))
            val y0 = Fixtures.vector(n, 3)
            val target = y0.copyOf()
            val y = DenseVector.wrap(target)
            val resets = resets(case)
            arm(EngineImplementation.Symv, listOf(a), emptyList(), timing(case)) {
                if (resets) y0.copyInto(target); blas.symv(alpha, a, x, beta, y, lower); target[0]
            }
        }

        "ger" -> {
            val m = d[0]
            val n = d[1]
            val original = Fixtures.matrix(m, n, 1)
            val target = Fixtures.matrix(m, n, 1)
            val a = target
            val x = DenseVector.wrap(Fixtures.vector(m, 2))
            val y = DenseVector.wrap(Fixtures.vector(n, 3))
            val resets = resets(case)
            // Without the reset, alternating the sign returns the matrix close to where it started.
            var signed = -alpha
            arm(EngineImplementation.Ger, listOf(a), emptyList(), timing(case)) {
                if (resets) original.values.copyInto(target.values) else signed = -signed
                blas.ger(if (resets) alpha else signed, x, y, a); target.values[0]
            }
        }

        "ger-shrinking" -> {
            val fixture = ShrinkingRankUpdate(d[0])
            val x = fixture.x.map { DenseVector.wrap(it) }
            val y = fixture.y.map { DenseVector.wrap(it) }
            val run = {
                for (step in x.indices) blas.ger(-1.0, x[step], y[step], fixture.matrix)
                fixture.matrix.values.last()
            }
            val choice = arm(EngineImplementation.Ger, listOf(fixture.matrix), x + y, "arithmetic", run)
            if (choice.work != null) fixture.verify(case.id, run)
            choice
        }

        "syr", "syr2" -> {
            val n = d[0]
            val original = Fixtures.matrix(n, n, 1)
            val target = Fixtures.matrix(n, n, 1)
            val a = target
            val x = DenseVector.wrap(Fixtures.vector(n, 2))
            val y = DenseVector.wrap(Fixtures.vector(n, 3))
            val operation = if (case.operation == "syr") EngineImplementation.Syr else EngineImplementation.Syr2
            val resets = resets(case)
            var signed = -alpha
            arm(operation, listOf(a), emptyList(), timing(case)) {
                if (resets) original.values.copyInto(target.values) else signed = -signed
                val scale = if (resets) alpha else signed
                if (operation == EngineImplementation.Syr) blas.syr(scale, x, a, lower) else blas.syr2(scale, x, y, a, lower)
                // The same entry the portable arm returns. Both triangles contain (0, 0), so it is written
                // either way, and reading a different corner per triangle would compare two different numbers.
                target.values[0]
            }
        }

        "trsv", "trmv" -> {
            val n = d[0]
            val a = triangle(n, 1)
            val x0 = Fixtures.vector(n, 2)
            val values = x0.copyOf()
            val x = DenseVector.wrap(values)
            val operation = if (case.operation == "trsv") EngineImplementation.Trsv else EngineImplementation.Trmv
            arm(operation, listOf(a), emptyList(), "reset-and-arithmetic") {
                x0.copyInto(values)
                if (operation == EngineImplementation.Trsv) blas.trsv(a, x, lower, transA, unit) else blas.trmv(a, x, lower, transA, unit)
                values[0]
            }
        }

        "gemm" -> {
            val (m, n, k) = d
            val left = Fixtures.matrix(if (transA) k else m, if (transA) m else k, 1)
            val rightMatrix = Fixtures.matrix(if (transB) n else k, if (transB) k else n, 2)
            val a = left
            val b = rightMatrix
            val original = Fixtures.matrix(m, n, 3)
            val target = Fixtures.matrix(m, n, 3)
            val c = target
            arm(EngineImplementation.Gemm, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.values.copyInto(target.values); blas.gemm(alpha, a, transA, b, transB, beta,
                    c); target.values[0]
            }
        }

        "symm" -> {
            val m = d[0]
            val n = d[1]
            val order = if (right) n else m
            val a = symmetric(order, 1)
            val b = Fixtures.matrix(m, n, 2)
            val original = Fixtures.matrix(m, n, 3)
            val target = Fixtures.matrix(m, n, 3)
            val c = target
            arm(EngineImplementation.Symm, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.values.copyInto(target.values); blas.symm(alpha, a, b, beta, c, lower, right); target.values[0]
            }
        }

        "gemmt" -> {
            val n = d[0]
            val k = d[1]
            val left = Fixtures.matrix(if (transA) k else n, if (transA) n else k, 1)
            val rightMatrix = Fixtures.matrix(if (transB) n else k, if (transB) k else n, 2)
            val a = left
            val b = rightMatrix
            val original = Fixtures.matrix(n, n, 3)
            val target = Fixtures.matrix(n, n, 3)
            val c = target
            arm(EngineImplementation.Gemmt, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.values.copyInto(target.values); blas.gemmt(alpha, a, transA, b, transB, beta, c, lower); target.values[0]
            }
        }

        "syrk", "syr2k" -> {
            val n = d[0]
            val k = d[1]
            val source = Fixtures.matrix(if (transA) k else n, if (transA) n else k, 1)
            val second = Fixtures.matrix(source.rows, source.cols, 2)
            val a = source
            val b = second
            val original = Fixtures.matrix(n, n, 3)
            val target = Fixtures.matrix(n, n, 3)
            val c = target
            val single = case.operation == "syrk"
            val operation = if (single) EngineImplementation.Syrk else EngineImplementation.Syr2k
            val operands = if (single) listOf(a, c) else listOf(a, b, c)
            arm(operation, operands, emptyList(), "reset-and-arithmetic") {
                original.values.copyInto(target.values)
                if (single) {
                    blas.syrk(alpha, a, transA, beta, c, lower)
                } else {
                    blas.syr2k(alpha, a, b, transA, beta, c, lower)
                }
                target.values[0]
            }
        }

        "trsm", "trmm" -> {
            val m = d[0]
            val n = d[1]
            val a = triangle(if (right) n else m, 1)
            val original = Fixtures.matrix(m, n, 2)
            val target = Fixtures.matrix(m, n, 2)
            val b = target
            val solve = case.operation == "trsm"
            val operation = if (solve) EngineImplementation.Trsm else EngineImplementation.Trmm
            arm(operation, listOf(a, b), emptyList(), "reset-and-arithmetic") {
                original.values.copyInto(target.values)
                if (solve) {
                    blas.trsm(a, b, lower, transA, unit, right, alpha)
                } else {
                    blas.trmm(a, b, lower, transA, unit, right, alpha)
                }
                target.values[0]
            }
        }

        else -> declined(case.operation)
    }
}
