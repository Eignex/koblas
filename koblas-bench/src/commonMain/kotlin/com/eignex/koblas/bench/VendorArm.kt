package com.eignex.koblas.bench

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import com.eignex.koblas.vendor.CallRoute
import com.eignex.koblas.vendor.Vendor
import com.eignex.koblas.vendor.VendorBlas
import com.eignex.koblas.vendor.VendorOperation
import com.eignex.koblas.vendor.exactArmRejection
import com.eignex.koblas.vendor.openVendorBlas

/**
 * The work for one vendor benchmark case, or the reason there is none to time.
 *
 * A rejection is a result, not a failure. An arm that cannot run a case honestly is expected to say so and
 * leave the row without a timing, because the alternative is a number that answers a different question than
 * the one the case asks.
 */
internal class VendorArm(val work: CaseWork?, val reason: String?)

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
internal fun vendorArm(case: BenchCase, blas: VendorBlas): VendorArm {
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

    fun triangle(order: Int, operand: Int): MatrixWindow {
        val source = Fixtures.triangular(order, operand, lower)
        val stored = when {
            unit && lower -> MatrixStructure.UnitLower
            unit -> MatrixStructure.UnitUpper
            lower -> MatrixStructure.TriangularLower
            else -> MatrixStructure.TriangularUpper
        }
        val window = MatrixWindow(source.data, order, order, structure = stored)
        return if (transA) window.transpose() else window
    }

    fun arm(operation: VendorOperation, matrices: List<MatrixWindow>, timing: String, run: () -> Double): VendorArm {
        exactArmRejection(blas, operation, matrices)?.let { return VendorArm(null, it) }
        val route = blas.routeOf(operation, matrices)
        return VendorArm(CaseWork(route.kind.name.lowercase(), timing, run, route = route), null)
    }

    fun declined(what: String) = VendorArm(null, "$what has no vendor entry point in the bound CBLAS surface")

    return when (case.operation) {
        "dot" -> {
            val x = VectorWindow(Fixtures.vector(d[0], 1), d[0])
            val y = VectorWindow(Fixtures.vector(d[0], 2), d[0])
            arm(VendorOperation.Dot, emptyList(), "arithmetic") { blas.dot(x, y) }
        }

        "axpy" -> {
            val x = VectorWindow(Fixtures.vector(d[0], 1), d[0])
            val initial = Fixtures.vector(d[0], 2)
            val target = initial.copyOf()
            val y = VectorWindow(target, d[0])
            arm(VendorOperation.Axpy, emptyList(), "reset-and-arithmetic") {
                initial.copyInto(target); blas.axpy(alpha, x, y); target[0]
            }
        }

        "scal" -> {
            val initial = Fixtures.vector(d[0], 1)
            val values = initial.copyOf()
            val x = VectorWindow(values, d[0])
            if (case.option("timing", "reset-and-arithmetic") == "arithmetic") {
                // Negation keeps the magnitudes normal over arbitrarily many timed invocations.
                arm(VendorOperation.Scal, emptyList(), "arithmetic") {
                    blas.scal(-1.0, x); values[0] + values[values.size - 1]
                }
            } else {
                arm(VendorOperation.Scal, emptyList(), "reset-and-arithmetic") {
                    initial.copyInto(values); blas.scal(alpha, x); values[0] + values[values.size - 1]
                }
            }
        }

        "nrm2" -> {
            val x = VectorWindow(Fixtures.vector(d[0], 1), d[0])
            arm(VendorOperation.Nrm2, emptyList(), "arithmetic") { blas.nrm2(x) }
        }

        "asum" -> {
            val x = VectorWindow(Fixtures.vector(d[0], 1), d[0])
            arm(VendorOperation.Asum, emptyList(), "arithmetic") { blas.asum(x) }
        }

        "iamax" -> {
            val x = VectorWindow(Fixtures.vector(d[0], 1), d[0])
            arm(VendorOperation.Iamax, emptyList(), "arithmetic") { blas.iamax(x).toDouble() }
        }

        "swap" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = VectorWindow(xs, d[0])
            val y = VectorWindow(ys, d[0])
            arm(VendorOperation.Swap, emptyList(), "reset-and-arithmetic") {
                x0.copyInto(xs); y0.copyInto(ys); blas.swap(x, y); xs[0] + ys[0]
            }
        }

        "rot" -> {
            val x0 = Fixtures.vector(d[0], 1)
            val y0 = Fixtures.vector(d[0], 2)
            val xs = x0.copyOf()
            val ys = y0.copyOf()
            val x = VectorWindow(xs, d[0])
            val y = VectorWindow(ys, d[0])
            arm(VendorOperation.Rot, emptyList(), "reset-and-arithmetic") {
                x0.copyInto(xs); y0.copyInto(ys); blas.rot(x, y, 0.8, 0.6); xs[0] + ys[0]
            }
        }

        "gemv" -> {
            val m = d[0]
            val n = d[1]
            val source = Fixtures.matrix(if (transA) n else m, if (transA) m else n, 1)
            val plain = MatrixWindow(source.data, source.rows, source.cols)
            val a = if (transA) plain.transpose() else plain
            val x = VectorWindow(Fixtures.vector(n, 2), n)
            val y0 = Fixtures.vector(m, 3)
            val target = y0.copyOf()
            val y = VectorWindow(target, m)
            arm(VendorOperation.Gemv, listOf(a), "reset-and-arithmetic") {
                y0.copyInto(target); blas.gemv(alpha, a, x, beta, y); target[0]
            }
        }

        "symv" -> {
            val n = d[0]
            val a = MatrixWindow(symmetric(n, 1).data, n, n, structure = structure())
            val x = VectorWindow(Fixtures.vector(n, 2), n)
            val y0 = Fixtures.vector(n, 3)
            val target = y0.copyOf()
            val y = VectorWindow(target, n)
            arm(VendorOperation.Symv, listOf(a), "reset-and-arithmetic") {
                y0.copyInto(target); blas.symv(alpha, a, x, beta, y); target[0]
            }
        }

        "ger" -> {
            val m = d[0]
            val n = d[1]
            val original = Fixtures.matrix(m, n, 1)
            val target = Fixtures.matrix(m, n, 1)
            val a = MatrixWindow(target.data, m, n)
            val x = VectorWindow(Fixtures.vector(m, 2), m)
            val y = VectorWindow(Fixtures.vector(n, 3), n)
            arm(VendorOperation.Ger, listOf(a), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.ger(alpha, x, y, a); target.data[0]
            }
        }

        "syr", "syr2" -> {
            val n = d[0]
            val original = Fixtures.matrix(n, n, 1)
            val target = Fixtures.matrix(n, n, 1)
            val a = MatrixWindow(target.data, n, n, structure = structure())
            val x = VectorWindow(Fixtures.vector(n, 2), n)
            val y = VectorWindow(Fixtures.vector(n, 3), n)
            val operation = if (case.operation == "syr") VendorOperation.Syr else VendorOperation.Syr2
            val corner = if (lower) 0 else n * n - 1
            arm(operation, listOf(a), "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (operation == VendorOperation.Syr) blas.syr(alpha, x, a) else blas.syr2(alpha, x, y, a)
                target.data[corner]
            }
        }

        "trsv", "trmv" -> {
            val n = d[0]
            val a = triangle(n, 1)
            val x0 = Fixtures.vector(n, 2)
            val values = x0.copyOf()
            val x = VectorWindow(values, n)
            val operation = if (case.operation == "trsv") VendorOperation.Trsv else VendorOperation.Trmv
            arm(operation, listOf(a), "reset-and-arithmetic") {
                x0.copyInto(values)
                if (operation == VendorOperation.Trsv) blas.trsv(a, x) else blas.trmv(a, x)
                values[0]
            }
        }

        "gemm" -> {
            val (m, n, k) = d
            val left = Fixtures.matrix(if (transA) k else m, if (transA) m else k, 1)
            val rightMatrix = Fixtures.matrix(if (transB) n else k, if (transB) k else n, 2)
            val a = MatrixWindow(left.data, left.rows, left.cols).let { if (transA) it.transpose() else it }
            val b = MatrixWindow(rightMatrix.data, rightMatrix.rows, rightMatrix.cols)
                .let { if (transB) it.transpose() else it }
            val original = Fixtures.matrix(m, n, 3)
            val target = Fixtures.matrix(m, n, 3)
            val c = MatrixWindow(target.data, m, n)
            arm(VendorOperation.Gemm, listOf(a, b, c), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.gemm(alpha, a, b, beta, c); target.data[0]
            }
        }

        "symm" -> {
            val m = d[0]
            val n = d[1]
            val order = if (right) n else m
            val a = MatrixWindow(symmetric(order, 1).data, order, order, structure = structure())
            val source = Fixtures.matrix(m, n, 2)
            val b = MatrixWindow(source.data, m, n)
            val original = Fixtures.matrix(m, n, 3)
            val target = Fixtures.matrix(m, n, 3)
            val c = MatrixWindow(target.data, m, n)
            arm(VendorOperation.Symm, listOf(a, b, c), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.symm(alpha, a, b, beta, c, right); target.data[0]
            }
        }

        "gemmt" -> {
            val n = d[0]
            val k = d[1]
            val left = Fixtures.matrix(if (transA) k else n, if (transA) n else k, 1)
            val rightMatrix = Fixtures.matrix(if (transB) n else k, if (transB) k else n, 2)
            val a = MatrixWindow(left.data, left.rows, left.cols).let { if (transA) it.transpose() else it }
            val b = MatrixWindow(rightMatrix.data, rightMatrix.rows, rightMatrix.cols)
                .let { if (transB) it.transpose() else it }
            val original = Fixtures.matrix(n, n, 3)
            val target = Fixtures.matrix(n, n, 3)
            val c = MatrixWindow(target.data, n, n, structure = structure())
            arm(VendorOperation.Gemmt, listOf(a, b, c), "reset-and-arithmetic") {
                original.data.copyInto(target.data); blas.gemmt(alpha, a, b, beta, c); target.data[0]
            }
        }

        "syrk", "syr2k" -> {
            val n = d[0]
            val k = d[1]
            val source = Fixtures.matrix(if (transA) k else n, if (transA) n else k, 1)
            val second = Fixtures.matrix(source.rows, source.cols, 2)
            val a = MatrixWindow(source.data, source.rows, source.cols).let { if (transA) it.transpose() else it }
            val b = MatrixWindow(second.data, second.rows, second.cols).let { if (transA) it.transpose() else it }
            val original = Fixtures.matrix(n, n, 3)
            val target = Fixtures.matrix(n, n, 3)
            val c = MatrixWindow(target.data, n, n, structure = structure())
            val single = case.operation == "syrk"
            val operation = if (single) VendorOperation.Syrk else VendorOperation.Syr2k
            val operands = if (single) listOf(a, c) else listOf(a, b, c)
            arm(operation, operands, "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (single) blas.syrk(alpha, a, beta, c) else blas.syr2k(alpha, a, b, beta, c)
                target.data[0]
            }
        }

        "trsm", "trmm" -> {
            val m = d[0]
            val n = d[1]
            val a = triangle(if (right) n else m, 1)
            val original = Fixtures.matrix(m, n, 2)
            val target = Fixtures.matrix(m, n, 2)
            val b = MatrixWindow(target.data, m, n)
            val solve = case.operation == "trsm"
            val operation = if (solve) VendorOperation.Trsm else VendorOperation.Trmm
            arm(operation, listOf(a, b), "reset-and-arithmetic") {
                original.data.copyInto(target.data)
                if (solve) blas.trsm(alpha, a, b, right) else blas.trmm(alpha, a, b, right)
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
