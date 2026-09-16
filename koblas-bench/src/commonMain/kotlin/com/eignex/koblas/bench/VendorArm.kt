package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.VendorBlas
import com.eignex.koblas.vendor.VendorOperation
import com.eignex.koblas.vendor.exactArmRejection
import com.eignex.koblas.vendor.openVendorBlas

/**
 * The work for one benchmark case on one arm, or the reason there is none to time.
 *
 * A rejection is a result, not a failure. An arm that cannot run a case honestly is expected to say so and
 * leave the row without a timing, because the alternative is a number that answers a different question than
 * the one the case asks.
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
 * The prefix is what keeps a mode from being run by the wrong entry point. Both runtimes name the same
 * vendors, so a check that only asked whether a mode named one would let the Native runner accept a
 * `jvm-vendor-` mode and report its rows under a runtime that never ran them.
 */
internal fun vendorForRuntime(mode: String, prefix: String): Vendor? =
    if (mode.startsWith(prefix)) vendorFromMode(mode) else null

/**
 * Opens the vendor a mode names, failing loudly rather than quietly measuring something else.
 *
 * The description carries the resolved file, the library's own version, and whether its single compute thread
 * was confirmed against it. A library that would not hold to one thread never opens, so reaching this point at
 * all is part of the evidence; recording which of the two it was keeps a report from claiming a check that a
 * library with no thread-count entry point cannot support.
 */
internal fun openVendorForMode(mode: String): Pair<VendorBlas, String> {
    val vendor = vendorFromMode(mode) ?: error("$mode is not a vendor mode")
    val blas = openVendorBlas(vendor)
        ?: error("requested vendor ${vendor.vendorName} is not installed; it cannot be benchmarked on this host")
    val description = "$mode/${blas.vendor.vendorName}/${blas.libraryPath}/" +
        "${blas.version}/threads=${blas.threadEvidence.label}"
    return blas to description
}

/**
 * The work for [case] through [blas], or a reason it is not an admissible measurement.
 *
 * Two things have to hold before a case is timed. The binding has to be able to run the operation over these
 * operands as a direct vendor call, which [exactArmRejection] decides; and the case has to be one the CBLAS
 * surface covers at all. Level 1 extensions like `sum` and `ssqd`, the fused panel kernels, the packed and tile
 * cases, and everything sparse have no vendor entry point, so they are declined here rather than quietly
 * measured through a Kotlin substitute wearing a vendor label.
 *
 * Fixtures, scalars and timing boundaries are the ones the other arms use, because a comparison between arms is
 * only a comparison if the work either side is the same.
 */
@Suppress("LongMethod", "CyclomaticComplexMethod") // one branch per benchmarked operation
internal fun vendorArm(case: BenchCase, blas: VendorBlas): ArmChoice {
    val d = case.dimensions
    val alpha = 0.875
    val beta = -0.25
    val lower = case.option("uplo", "L") == "L"
    val transA = case.flag("transA")
    val transB = case.flag("transB")
    val unit = case.option("diag", "N") == "U"
    val right = case.option("side", "L") == "R"

    fun symmetric(order: Int, operand: Int): DenseMatrix = Fixtures.matrix(order, order, operand)
    fun structure(): MatrixStructure =
        if (lower) MatrixStructure.SymmetricLower else MatrixStructure.SymmetricUpper

    fun triangle(order: Int, operand: Int): DenseMatrix = Fixtures.triangular(order, operand, lower)

    /** The triangle and diagonal the case's flags declare, which travels beside the operand rather than in it. */
    fun triangleStructure(): MatrixStructure = when {
        unit && lower -> MatrixStructure.UnitLower
        unit -> MatrixStructure.UnitUpper
        lower -> MatrixStructure.TriangularLower
        else -> MatrixStructure.TriangularUpper
    }

    fun arm(
        operation: VendorOperation,
        matrices: List<DenseMatrix>,
        vectors: List<DenseVector>,
        timing: String,
        run: () -> Double,
    ): ArmChoice {
        // Both operand lists reach the route, so a case whose operands make the call no-work is described that
        // way and declined rather than timed as vendor arithmetic it never performed.
        exactArmRejection(blas, operation, matrices, vectors)?.let { return ArmChoice(null, it) }
        val route = blas.routeOf(operation, matrices, vectors)
        return ArmChoice(CaseWork(route.kind.name.lowercase(), timing, run, kernel = vendorKernel(route)), null)
    }

    fun declined(what: String) = ArmChoice(null, "$what has no vendor entry point in the bound CBLAS surface")

    return when (case.operation) {
        "dot" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            val y = DenseVector.wrap(Fixtures.vector(d[0], 2))
            arm(VendorOperation.Dot, emptyList(), listOf(x, y), "arithmetic") { blas.dot(x, y) }
        }

        "axpy" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            val initial = Fixtures.vector(d[0], 2)
            val target = initial.copyOf()
            val y = DenseVector.wrap(target)
            arm(VendorOperation.Axpy, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                initial.copyInto(target); blas.axpy(alpha, x, y); target[0]
            }
        }

        "scal" -> {
            val initial = Fixtures.vector(d[0], 1)
            val values = initial.copyOf()
            val x = DenseVector.wrap(values)
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Negation keeps the magnitudes normal over arbitrarily many timed invocations.
                arm(VendorOperation.Scal, emptyList(), listOf(x), "arithmetic") {
                    blas.scal(-1.0, x); values[0] + values[values.size - 1]
                }
            } else {
                arm(VendorOperation.Scal, emptyList(), listOf(x), "reset-and-arithmetic") {
                    initial.copyInto(values); blas.scal(alpha, x); values[0] + values[values.size - 1]
                }
            }
        }

        "nrm2" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(VendorOperation.Nrm2, emptyList(), listOf(x), "arithmetic") { blas.nrm2(x) }
        }

        "asum" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(VendorOperation.Asum, emptyList(), listOf(x), "arithmetic") { blas.asum(x) }
        }

        "iamax" -> {
            val x = DenseVector.wrap(Fixtures.vector(d[0], 1))
            arm(VendorOperation.Iamax, emptyList(), listOf(x), "arithmetic") { blas.iamax(x).toDouble() }
        }

        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = DenseVector.wrap(xs)
            val y = DenseVector.wrap(ys)
            arm(VendorOperation.Swap, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                x0.copyInto(xs); y0.copyInto(ys); blas.swap(x, y); xs[0] + ys[0]
            }
        }

        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = DenseVector.wrap(xs)
            val y = DenseVector.wrap(ys)
            arm(VendorOperation.Rot, emptyList(), listOf(x, y), "reset-and-arithmetic") {
                x0.copyInto(xs); y0.copyInto(ys); blas.rot(x, y, 0.8, 0.6); xs[0] + ys[0]
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
            arm(VendorOperation.Gemv, listOf(a), emptyList(), "reset-and-arithmetic") {
                y0.copyInto(target); blas.gemv(alpha, a, transA, x, beta, y); target[0]
            }
        }

        "symv" -> {
            val n = d[0]
            val a = symmetric(n, 1)
            val x = DenseVector.wrap(Fixtures.vector(n, 2))
            val y0 = Fixtures.vector(n, 3)
            val target = y0.copyOf()
            val y = DenseVector.wrap(target)
            arm(VendorOperation.Symv, listOf(a), emptyList(), "reset-and-arithmetic") {
                y0.copyInto(target); blas.symv(alpha, a, structure(), x, beta, y); target[0]
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
            arm(VendorOperation.Ger, listOf(a), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.ger(alpha, x, y, a); target.data[0]
            }
        }

        "syr", "syr2" -> {
            val n = d[0]
            val original = Fixtures.matrix(n, n, 1)
            val target = Fixtures.matrix(n, n, 1)
            val a = target
            val x = DenseVector.wrap(Fixtures.vector(n, 2))
            val y = DenseVector.wrap(Fixtures.vector(n, 3))
            val operation = if (case.operation == "syr") VendorOperation.Syr else VendorOperation.Syr2
            arm(operation, listOf(a), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (operation == VendorOperation.Syr) blas.syr(alpha, x, a, structure()) else blas.syr2(alpha, x, y, a, structure())
                // The same entry the portable arm returns. Both triangles contain (0, 0), so it is written
                // either way, and reading a different corner per triangle would compare two different numbers.
                target.data[0]
            }
        }

        "trsv", "trmv" -> {
            val n = d[0]
            val a = triangle(n, 1)
            val x0 = Fixtures.vector(n, 2)
            val values = x0.copyOf()
            val x = DenseVector.wrap(values)
            val operation = if (case.operation == "trsv") VendorOperation.Trsv else VendorOperation.Trmv
            arm(operation, listOf(a), emptyList(), "reset-and-arithmetic") {
                x0.copyInto(values)
                if (operation == VendorOperation.Trsv) blas.trsv(a, triangleStructure(), transA, x) else blas.trmv(a, triangleStructure(), transA, x)
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
            arm(VendorOperation.Gemm, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.gemm(alpha, a, transA, b, transB, beta, c); target.data[0]
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
            arm(VendorOperation.Symm, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.symm(alpha, a, structure(), b, beta, c, right); target.data[0]
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
            arm(VendorOperation.Gemmt, listOf(a, b, c), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.gemmt(alpha, a, transA, b, transB, beta, c, structure()); target.data[0]
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
            val operation = if (single) VendorOperation.Syrk else VendorOperation.Syr2k
            val operands = if (single) listOf(a, c) else listOf(a, b, c)
            arm(operation, operands, emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (single) {
                    blas.syrk(alpha, a, transA, beta, c, structure())
                } else {
                    blas.syr2k(alpha, a, b, transA, beta, c, structure())
                }
                target.data[0]
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
            val operation = if (solve) VendorOperation.Trsm else VendorOperation.Trmm
            arm(operation, listOf(a, b), emptyList(), "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (solve) {
                    blas.trsm(alpha, a, triangleStructure(), transA, b, right)
                } else {
                    blas.trmm(alpha, a, triangleStructure(), transA, b, right)
                }
                target.data[0]
            }
        }

        else -> declined(case.operation)
    }
}

/** The attribution a vendor row carries, taken from the binding rather than rebuilt from the mode string. */
internal fun vendorKernel(route: CallRoute): String = buildString {
    append(route.vendor?.vendorName ?: "none")
    append('/')
    append(route.entryPoint ?: route.adapter ?: route.kind.name.lowercase())
}
