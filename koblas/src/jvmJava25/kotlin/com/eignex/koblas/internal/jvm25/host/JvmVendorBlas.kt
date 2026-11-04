@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.MatrixStructure
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.host.Cblas
import com.eignex.koblas.internal.host.HostBlas
import com.eignex.koblas.internal.host.diagFor
import com.eignex.koblas.internal.host.positiveIncrement
import com.eignex.koblas.internal.host.sideFor
import com.eignex.koblas.internal.host.transposeFor
import com.eignex.koblas.internal.host.uploFor
import com.eignex.koblas.sparse.*
import java.lang.foreign.Arena
import java.lang.invoke.MethodHandle

/**
 * A vendor BLAS reached through `java.lang.foreign`.
 *
 * Handles are bound once at construction. Each call opens a confined arena, copies its operands in, makes one
 * non-critical downcall, copies its outputs back, and closes the arena on the way out including on an
 * exception. Nothing is cached between calls and no segment escapes one, so instances are safe to share and
 * concurrent calls on one instance do not interfere.
 */
internal class JvmVendorBlas(private val library: JvmVendorLibrary) : HostBlas() {
    override val name: String get() = vendor.vendorName.lowercase()
    override fun minimumLength(implementation: EngineImplementation): Int = 1

    override val vendor: Vendor get() = library.vendor
    override val libraryPath: String get() = library.resolvedFile
    override val version: String get() = library.version
    override val threadEvidence: ThreadEvidence get() = library.threadEvidence

    /** Bound by [EngineImplementation.ordinal], null where the library does not export the entry point. */
    private val handles: Array<MethodHandle?> = Array(EngineImplementation.entries.size) {
        val operation = EngineImplementation.entries[it]
        descriptorFor(operation)?.let { descriptor -> library.handleOrNull(operation.entryPoint, descriptor) }
    }

    override val implementations: Set<EngineImplementation> =
        EngineImplementation.entries.filterTo(LinkedHashSet()) { handles[it.ordinal] != null }

    private fun handle(operation: EngineImplementation): MethodHandle {
        val exported = handles[operation.ordinal] ?: throw UnsupportedOperationException(
            "${operation.entryPoint} is not exported by ${vendor.vendorName}",
        )
        library.holdCallingThread()
        return exported
    }

    // Level 1 bindings stage foreign memory while keeping arithmetic in this library.

    override fun dot(
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ): Double {
        val x = StridedVector(a, aOff, len, aStride)
        val y = StridedVector(b, bOff, len, bStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            handle(EngineImplementation.Dot)
                .invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        reduce(EngineImplementation.Nrm2, v, vOff, len, vStride)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        reduce(EngineImplementation.Asum, v, vOff, len, vStride)

    private fun reduce(operation: EngineImplementation, v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            handle(operation).invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Double
        }
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return -1
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val found = handle(EngineImplementation.Iamax)
                .invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Int
            if (x.stride >= 0) found else x.size - 1 - found
        }
    }

    override fun axpy(
        y: DoubleArray,
        yOff: Int,
        alpha: Double,
        x: DoubleArray,
        xOff: Int,
        len: Int,
        yStride: Int,
        xStride: Int,
    ) {
        val source = StridedVector(x, xOff, len, xStride)
        val destination = StridedVector(y, yOff, len, yStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(source)
            val ny = arena.stage(destination)
            handle(EngineImplementation.Axpy)
                .invokeExact(source.size, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            handle(EngineImplementation.Scale)
                .invokeExact(x.size, alpha, nx.segment, positiveIncrement(nx.increment, x.size))
            nx.writeBack()
        }
    }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        pair(EngineImplementation.Copy, a, aOff, b, bOff, len, aStride, bStride, exchange = false)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        pair(EngineImplementation.Swap, a, aOff, b, bOff, len, aStride, bStride, exchange = true)

    /** `cblas_dcopy` or `cblas_dswap`, where [exchange] also writes the first operand back. */
    private fun pair(
        operation: EngineImplementation,
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
        exchange: Boolean,
    ) {
        val x = StridedVector(a, aOff, len, aStride)
        val y = StridedVector(b, bOff, len, bStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            handle(operation).invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            if (exchange) nx.writeBack()
            ny.writeBack()
        }
    }

    override fun rot(
        x: DoubleArray,
        xOff: Int,
        y: DoubleArray,
        yOff: Int,
        len: Int,
        c: Double,
        s: Double,
        xStride: Int,
        yStride: Int,
    ) {
        val source = StridedVector(x, xOff, len, xStride)
        val destination = StridedVector(y, yOff, len, yStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(source)
            val ny = arena.stage(destination)
            handle(EngineImplementation.Rot)
                .invokeExact(source.size, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    override fun cblasGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y, read = beta != 0.0)
            handle(EngineImplementation.Gemv).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun cblasSymv(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            val ny = arena.stage(y, read = beta != 0.0)
            handle(EngineImplementation.Symv).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun cblasGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            handle(EngineImplementation.Ger).invokeExact(
                Cblas.COL_MAJOR, a.rows, a.cols, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun cblasSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            handle(EngineImplementation.Syr).invokeExact(
                Cblas.COL_MAJOR,
                uploFor(structure),
                a.rows,
                alpha,
                nx.segment,
                nx.increment,
                na.segment,
                na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun cblasSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            handle(EngineImplementation.Syr2).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            handle(operation).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA), diagFor(structure),
                a.rows, na.segment, na.leadingDimension, nx.segment, nx.increment,
            )
            nx.writeBack()
        }
    }

    // Level 3. Every operand is contiguous column-major, so the layout is fixed and a transpose is a flag.

    override fun cblasGemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c, read = beta != 0.0)
            handle(EngineImplementation.Gemm).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                c.rows, c.cols, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSymm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        rightSide: Boolean,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nb = arena.stage(b)
            val nc = arena.stage(c, read = beta != 0.0)
            handle(EngineImplementation.Symm).invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSyrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nc = arena.stage(c, structure, read = beta != 0.0)
            handle(EngineImplementation.Syrk).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSyr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c, structure, read = beta != 0.0)
            handle(EngineImplementation.Syr2k).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasTriangularMatrix(
        operation: EngineImplementation,
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nb = arena.stage(b)
            handle(operation).invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure),
                transposeFor(transposeA), diagFor(structure), b.rows, b.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension,
            )
            nb.writeBack()
        }
    }

    /** Calls only this library's exported `cblas_dgemmt`. */
    override fun cblasGemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c, structure, read = beta != 0.0)
            handle(EngineImplementation.Gemmt).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                transposeFor(transposeB), c.rows, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }
}
