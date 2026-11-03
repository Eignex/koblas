@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.host.Cblas
import com.eignex.koblas.internal.host.diagFor
import com.eignex.koblas.internal.host.exportedImplementations
import com.eignex.koblas.internal.host.positiveIncrement
import com.eignex.koblas.internal.host.sideFor
import com.eignex.koblas.internal.host.transposeFor
import com.eignex.koblas.internal.host.uploFor
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.sparse.*
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/**
 * A vendor BLAS reached through `java.lang.foreign`.
 *
 * Handles are bound once at construction. Each call opens a confined arena, copies its operands in, makes one
 * non-critical downcall, copies its outputs back, and closes the arena on the way out including on an
 * exception. Nothing is cached between calls and no segment escapes one, so instances are safe to share and
 * concurrent calls on one instance do not interfere.
 */
internal class JvmVendorBlas(private val library: JvmVendorLibrary) : EngineKernels {
    override val name: String get() = vendor.vendorName.lowercase()
    override fun minimumLength(implementation: EngineImplementation): Int = 1

    override val vendor: Vendor get() = library.vendor
    override val libraryPath: String get() = library.resolvedFile
    override val version: String get() = library.version
    override val threadEvidence: ThreadEvidence get() = library.threadEvidence

    override val implementations: Set<EngineImplementation> = exportedImplementations(library::exports)

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
            supplied(dotHandle).invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            supplied(nrm2Handle).invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Double
        }
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            supplied(asumHandle).invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Double
        }
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return -1
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val found = supplied(
                iamaxHandle,
            ).invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Int
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
            supplied(axpyHandle).invokeExact(source.size, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            supplied(scalHandle).invokeExact(x.size, alpha, nx.segment, positiveIncrement(nx.increment, x.size))
            nx.writeBack()
        }
    }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        val x = StridedVector(a, aOff, len, aStride)
        val y = StridedVector(b, bOff, len, bStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            supplied(copyHandle).invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        val x = StridedVector(a, aOff, len, aStride)
        val y = StridedVector(b, bOff, len, bStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            supplied(swapHandle).invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            nx.writeBack()
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
            supplied(rotHandle).invokeExact(source.size, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    private fun cblasGemv(
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
            supplied(gemvHandle).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    private fun cblasSymv(
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
            supplied(symvHandle).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    private fun cblasGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            supplied(gerHandle).invokeExact(
                Cblas.COL_MAJOR, a.rows, a.cols, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    private fun cblasSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            supplied(syrHandle).invokeExact(
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

    private fun cblasSyr2(
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
            supplied(syr2Handle).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    private fun cblasTrsv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, supplied(trsvHandle))

    private fun cblasTrmv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, supplied(trmvHandle))

    @Suppress("LongParameterList") // the shared triangular vector signature plus its handle
    private fun triangularVector(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
        handle: MethodHandle,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            handle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA), diagFor(structure),
                a.rows, na.segment, na.leadingDimension, nx.segment, nx.increment,
            )
            nx.writeBack()
        }
    }

    // Level 3. Every operand is contiguous column-major, so the layout is fixed and a transpose is a flag.

    @Suppress("LongParameterList") // the BLAS dgemm signature
    private fun cblasGemm(
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
            supplied(gemmHandle).invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                c.rows, c.cols, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature
    private fun cblasSymm(
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
            supplied(symmHandle).invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature
    private fun cblasSyrk(
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
            supplied(syrkHandle).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    private fun cblasSyr2k(
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
            supplied(syr2kHandle).invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    private fun cblasTrmm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, supplied(trmmHandle))

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    private fun cblasTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, supplied(trsmHandle))

    @Suppress("LongParameterList") // the shared triangular matrix signature plus its handle
    private fun triangularMatrix(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
        handle: MethodHandle,
    ) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nb = arena.stage(b)
            handle.invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure),
                transposeFor(transposeA), diagFor(structure), b.rows, b.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension,
            )
            nb.writeBack()
        }
    }

    /** Calls only this library's exported `cblas_dgemmt`. */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    private fun cblasGemmt(
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
            gemmtHandle().invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                transposeFor(transposeB), c.rows, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    // The routines run inside the shared preparation, so a call reaching this library directly is checked and
    // staged exactly as one routed here by a composed engine.

    private fun requireExport(operation: EngineImplementation) {
        if (operation !in implementations) {
            throw UnsupportedOperationException("${operation.entryPoint} is not exported by ${vendor.vendorName}")
        }
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemv)
        prepareGemv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, transpose, scratch?.arrays) {
                sa,
                sx,
                sxOffset,
                sxStride,
            ->
            val rows = if (transpose) a.cols else a.rows
            cblasGemv(
                alpha,
                sa,
                transpose,
                StridedVector(sx, sxOffset, if (transpose) a.rows else a.cols, sxStride),
                beta,
                StridedVector(y, yOffset, rows, yStride),
            )
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Symv)
        prepareSymv(this, alpha, a, x, xOffset, xStride, beta, y, yOffset, yStride, lower, scratch?.arrays) {
                sa,
                sx,
                sxOffset,
                sxStride,
            ->
            cblasSymv(
                alpha,
                sa,
                symmetricStructure(lower),
                StridedVector(sx, sxOffset, a.rows, sxStride),
                beta,
                StridedVector(y, yOffset, a.rows, yStride),
            )
        }
    }

    override fun ger(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Ger)
        prepareGer(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch?.arrays) { sx, sxo, sxs, sy, syo, sys ->
            cblasGer(alpha, StridedVector(sx, sxo, a.rows, sxs), StridedVector(sy, syo, a.cols, sys), a)
        }
    }

    override fun syr(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr)
        prepareSyr(alpha, x, xOffset, xStride, a, scratch?.arrays) { sx, sxo, sxs ->
            cblasSyr(alpha, StridedVector(sx, sxo, a.rows, sxs), a, symmetricStructure(lower))
        }
    }

    override fun syr2(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        a: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr2)
        prepareSyr2(alpha, x, xOffset, xStride, y, yOffset, yStride, a, scratch?.arrays) { sx, sxo, sxs, sy, syo, sys ->
            cblasSyr2(
                alpha,
                StridedVector(sx, sxo, a.rows, sxs),
                StridedVector(sy, syo, a.rows, sys),
                a,
                symmetricStructure(lower),
            )
        }
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trsv)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trsv", scratch?.arrays) { sa ->
            cblasTrsv(sa, triangle(lower, unitDiag), transpose, StridedVector(x, xOffset, a.rows, xStride))
        }
    }

    override fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trmv)
        prepareTriangularVector(a, x, xOffset, xStride, lower, unitDiag, "trmv", scratch?.arrays) { sa ->
            cblasTrmv(sa, triangle(lower, unitDiag), transpose, StridedVector(x, xOffset, a.rows, xStride))
        }
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemm)
        prepareGemm(alpha, a, transposeA, b, transposeB, beta, c, scratch?.arrays) { sa, sb, _ ->
            cblasGemm(alpha, sa, transposeA, sb, transposeB, beta, c)
        }
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Gemmt)
        prepareGemmt(alpha, a, transposeA, b, transposeB, beta, c, lower, scratch?.arrays) { sa, sb ->
            cblasGemmt(alpha, sa, transposeA, sb, transposeB, beta, c, symmetricStructure(lower))
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Symm)
        prepareSymm(this, alpha, a, b, beta, c, lower, right, scratch?.arrays) { sa, sb ->
            cblasSymm(alpha, sa, symmetricStructure(lower), sb, beta, c, right)
        }
    }

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syrk)
        prepareSyrk(alpha, a, transpose, beta, c, lower, scratch?.arrays) { sa ->
            cblasSyrk(alpha, sa, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Syr2k)
        prepareSyr2k(alpha, a, b, transpose, beta, c, lower, scratch?.arrays) { sa, sb ->
            cblasSyr2k(alpha, sa, sb, transpose, beta, c, symmetricStructure(lower))
        }
    }

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trsm)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trsm", scratch?.arrays) { sa ->
            cblasTrsm(alpha, sa, triangle(lower, unitDiag), transpose, b, right)
        }
    }

    override fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        scratch: Scratch?,
    ) {
        requireExport(EngineImplementation.Trmm)
        prepareTriangularMatrix(a, b, lower, unitDiag, right, alpha, "trmm", scratch?.arrays) { sa ->
            cblasTrmm(alpha, sa, triangle(lower, unitDiag), transpose, b, right)
        }
    }
    private fun gemmtHandle(): MethodHandle = supplied(gemmtOrNull)

    private fun supplied(handle: MethodHandle?): MethodHandle {
        val exported = handle ?: throw UnsupportedOperationException(
            "Operation is not exported by ${vendor.vendorName}",
        )
        library.holdCallingThread()
        return exported
    }

    private val dotHandle = library.handleOrNull(
        EngineImplementation.Dot.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val nrm2Handle = library.handleOrNull(
        EngineImplementation.Nrm2.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val asumHandle = library.handleOrNull(
        EngineImplementation.Asum.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )

    /**
     * `CBLAS_INDEX` is `size_t` in the reference header and in several vendors, and `int` in others. The low
     * 32 bits agree for every index a 32-bit BLAS integer can address, so binding it as `int` reads the same
     * value either way and avoids depending on which one this library used.
     */
    private val iamaxHandle = library.handleOrNull(
        EngineImplementation.Iamax.entryPoint,
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val axpyHandle = library.handleOrNull(
        EngineImplementation.Axpy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val scalHandle = library.handleOrNull(
        EngineImplementation.Scale.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT),
    )
    private val copyHandle = library.handleOrNull(
        EngineImplementation.Copy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val swapHandle = library.handleOrNull(
        EngineImplementation.Swap.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val rotHandle = library.handleOrNull(
        EngineImplementation.Rot.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE),
    )
    private val gemvHandle = library.handleOrNull(
        EngineImplementation.Gemv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symvHandle = library.handleOrNull(
        EngineImplementation.Symv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val gerHandle = library.handleOrNull(
        EngineImplementation.Ger.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val syrHandle = library.handleOrNull(
        EngineImplementation.Syr.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT,
            JAVA_INT,
            JAVA_INT,
            JAVA_DOUBLE,
            ADDRESS,
            JAVA_INT,
            ADDRESS,
            JAVA_INT,
        ),
    )
    private val syr2Handle = library.handleOrNull(
        EngineImplementation.Syr2.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val trsvHandle = library.handleOrNull(EngineImplementation.Trsv.entryPoint, TRIANGULAR_VECTOR)
    private val trmvHandle = library.handleOrNull(EngineImplementation.Trmv.entryPoint, TRIANGULAR_VECTOR)
    private val gemmHandle = library.handleOrNull(
        EngineImplementation.Gemm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symmHandle = library.handleOrNull(
        EngineImplementation.Symm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syrkHandle = library.handleOrNull(
        EngineImplementation.Syrk.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syr2kHandle = library.handleOrNull(
        EngineImplementation.Syr2k.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val trmmHandle = library.handleOrNull(EngineImplementation.Trmm.entryPoint, TRIANGULAR_MATRIX)
    private val trsmHandle = library.handleOrNull(EngineImplementation.Trsm.entryPoint, TRIANGULAR_MATRIX)
    private val gemmtOrNull = library.handleOrNull(
        EngineImplementation.Gemmt.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )

    private companion object {
        val TRIANGULAR_VECTOR: FunctionDescriptor = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        )
        val TRIANGULAR_MATRIX: FunctionDescriptor = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        )
    }
}
