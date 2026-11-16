@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.internal.host.*
import com.eignex.koblas.internal.operations.MatrixStructure
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
internal class JvmVendorBlas(private val library: JvmVendorLibrary) : HostBlas {
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
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(a, aOff, len, aStride)
            val ny = arena.stageVector(b, bOff, len, bStride)
            handle(EngineImplementation.Dot)
                .invokeExact(len, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        reduce(EngineImplementation.Nrm2, v, vOff, len, vStride)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        reduce(EngineImplementation.Asum, v, vOff, len, vStride)

    private fun reduce(operation: EngineImplementation, v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(v, vOff, len, vStride)
            handle(operation).invokeExact(len, nx.segment, positiveIncrement(nx.increment, len)) as Double
        }
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        if (len == 0) return -1
        return Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(v, vOff, len, vStride)
            val found = handle(EngineImplementation.Iamax)
                .invokeExact(len, nx.segment, positiveIncrement(nx.increment, len)) as Int
            if (vStride >= 0) found else len - 1 - found
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
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(x, xOff, len, xStride)
            val ny = arena.stageVector(y, yOff, len, yStride)
            handle(EngineImplementation.Axpy)
                .invokeExact(len, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(v, vOff, len, vStride)
            handle(EngineImplementation.Scale)
                .invokeExact(len, alpha, nx.segment, positiveIncrement(nx.increment, len))
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
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(a, aOff, len, aStride)
            val ny = arena.stageVector(b, bOff, len, bStride)
            handle(operation).invokeExact(len, nx.segment, nx.increment, ny.segment, ny.increment)
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
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stageVector(x, xOff, len, xStride)
            val ny = arena.stageVector(y, yOff, len, yStride)
            handle(EngineImplementation.Rot)
                .invokeExact(len, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    override fun cblasGemv(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            val ny = arena.stageVector(y, yOffset, ySize, yStride, read = beta != 0.0)
            handle(EngineImplementation.Gemv).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), aRows, aColumns, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun cblasSymv(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            val ny = arena.stageVector(y, yOffset, ySize, yStride, read = beta != 0.0)
            handle(EngineImplementation.Symv).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), aRows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun cblasGer(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            val ny = arena.stageVector(y, yOffset, ySize, yStride)
            handle(EngineImplementation.Ger).invokeExact(
                Cblas.COL_MAJOR, aRows, aColumns, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun cblasSyr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            handle(EngineImplementation.Syr).invokeExact(
                Cblas.COL_MAJOR,
                uploFor(structure),
                aRows,
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
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        ySize: Int,
        yStride: Int,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            val ny = arena.stageVector(y, yOffset, ySize, yStride)
            handle(EngineImplementation.Syr2).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), aRows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DoubleArray,
        xOffset: Int,
        xSize: Int,
        xStride: Int,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nx = arena.stageVector(x, xOffset, xSize, xStride)
            handle(operation).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA), diagFor(structure),
                aRows, na.segment, na.leadingDimension, nx.segment, nx.increment,
            )
            nx.writeBack()
        }
    }

    // Level 3. Every operand is contiguous column-major, so the layout is fixed and a transpose is a flag.

    override fun cblasGemm(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeB: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
    ) {
        val depth = if (transposeA) aRows else aColumns
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nb = arena.stageMatrix(b, bOffset, bRows, bColumns, bLeadingDimension)
            val nc = arena.stageMatrix(c, cOffset, cRows, cColumns, cLeadingDimension, read = beta != 0.0)
            handle(EngineImplementation.Gemm).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                cRows, cColumns, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSymm(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        rightSide: Boolean,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nb = arena.stageMatrix(b, bOffset, bRows, bColumns, bLeadingDimension)
            val nc = arena.stageMatrix(c, cOffset, cRows, cColumns, cLeadingDimension, read = beta != 0.0)
            handle(EngineImplementation.Symm).invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), cRows, cColumns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSyrk(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) aRows else aColumns
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nc = arena.stageMatrix(c, cOffset, cRows, cColumns, cLeadingDimension, structure, read = beta != 0.0)
            handle(EngineImplementation.Syrk).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                cRows, depth, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasSyr2k(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeA: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) aRows else aColumns
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nb = arena.stageMatrix(b, bOffset, bRows, bColumns, bLeadingDimension)
            val nc = arena.stageMatrix(c, cOffset, cRows, cColumns, cLeadingDimension, structure, read = beta != 0.0)
            handle(EngineImplementation.Syr2k).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                cRows, depth, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun cblasTriangularMatrix(
        operation: EngineImplementation,
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        rightSide: Boolean,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension, structure)
            val nb = arena.stageMatrix(b, bOffset, bRows, bColumns, bLeadingDimension)
            handle(operation).invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure),
                transposeFor(transposeA), diagFor(structure), bRows, bColumns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension,
            )
            nb.writeBack()
        }
    }

    /** Calls only this library's exported `cblas_dgemmt`. */
    override fun cblasGemmt(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        aRows: Int,
        aColumns: Int,
        aLeadingDimension: Int,
        transposeA: Boolean,
        b: DoubleArray,
        bOffset: Int,
        bRows: Int,
        bColumns: Int,
        bLeadingDimension: Int,
        transposeB: Boolean,
        beta: Double,
        c: DoubleArray,
        cOffset: Int,
        cRows: Int,
        cColumns: Int,
        cLeadingDimension: Int,
        structure: MatrixStructure,
    ) {
        val depth = if (transposeA) aRows else aColumns
        Arena.ofConfined().use { arena ->
            val na = arena.stageMatrix(a, aOffset, aRows, aColumns, aLeadingDimension)
            val nb = arena.stageMatrix(b, bOffset, bRows, bColumns, bLeadingDimension)
            val nc = arena.stageMatrix(c, cOffset, cRows, cColumns, cLeadingDimension, structure, read = beta != 0.0)
            handle(EngineImplementation.Gemmt).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                transposeFor(transposeB), cRows, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }
}
