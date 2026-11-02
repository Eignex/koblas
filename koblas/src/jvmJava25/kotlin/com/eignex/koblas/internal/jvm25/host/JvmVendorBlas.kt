@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.jvm25.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.internal.dense.triangularMatrix
import com.eignex.koblas.internal.host.Cblas
import com.eignex.koblas.internal.host.diagFor
import com.eignex.koblas.internal.host.positiveIncrement
import com.eignex.koblas.internal.host.sideFor
import com.eignex.koblas.internal.host.transposeFor
import com.eignex.koblas.internal.host.uploFor
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
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
internal class JvmVendorBlas(private val library: JvmVendorLibrary) : KoblasEngine() {
    override val name: String get() = vendor.vendorName.lowercase()
    override val minimumVectorLength: Int get() = 1

    override val vendor: Vendor get() = library.vendor
    override val libraryPath: String get() = library.resolvedFile
    override val version: String get() = library.version
    override val threadEvidence: ThreadEvidence get() = library.threadEvidence

    override val implementations: Set<EngineImplementation> = EngineImplementation.entries
        .filterTo(LinkedHashSet()) { it.entryPoint.isNotEmpty() && library.exports(it.entryPoint) }

    // Level 1. A vector window is always expressible, so these never stage and never compose.

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
            dotHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            nrm2Handle.invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Double
        }
    }

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            asumHandle.invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Double
        }
    }

    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return -1
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val found = iamaxHandle.invokeExact(x.size, nx.segment, positiveIncrement(nx.increment, x.size)) as Int
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
            axpyHandle.invokeExact(source.size, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        val x = StridedVector(v, vOff, len, vStride)
        if (len == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            scalHandle.invokeExact(x.size, alpha, nx.segment, positiveIncrement(nx.increment, x.size))
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
            copyHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
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
            swapHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
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
            rotHandle.invokeExact(source.size, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    override fun computeGemv(
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
            gemvHandle.invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun computeSymv(
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
            symvHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun computeGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            gerHandle.invokeExact(
                Cblas.COL_MAJOR, a.rows, a.cols, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun computeSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) {
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, structure)
            val nx = arena.stage(x)
            syrHandle.invokeExact(
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

    override fun computeSyr2(
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
            syr2Handle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun computeTrsv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, trsvHandle)

    override fun computeTrmv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, trmvHandle)

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
    override fun computeGemm(
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
            gemmHandle.invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                c.rows, c.cols, depth, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsymm signature
    override fun computeSymm(
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
            symmHandle.invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature
    override fun computeSyrk(
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
            syrkHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    override fun computeSyr2k(
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
            syr2kHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun computeTrmm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, trmmHandle)

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun computeTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, trsmHandle)

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

    /** Direct where the vendor exports `cblas_dgemmt`, and otherwise the shared composition. */
    @Suppress("LongParameterList") // the BLAS gemmt signature
    override fun computeGemmt(
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
        if (EngineImplementation.Gemmt !in implementations) {
            error("Backend does not supply Gemmt")
        }
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

    private fun gemmtHandle(): MethodHandle = checkNotNull(gemmtOrNull) { "gemmt is not exported" }

    private val dotHandle = library.handle(
        EngineImplementation.Dot.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val nrm2Handle = library.handle(
        EngineImplementation.Nrm2.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val asumHandle = library.handle(
        EngineImplementation.Asum.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )

    /**
     * `CBLAS_INDEX` is `size_t` in the reference header and in several vendors, and `int` in others. The low
     * 32 bits agree for every index a 32-bit BLAS integer can address, so binding it as `int` reads the same
     * value either way and avoids depending on which one this library used.
     */
    private val iamaxHandle = library.handle(
        EngineImplementation.Iamax.entryPoint,
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val axpyHandle = library.handle(
        EngineImplementation.Axpy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val scalHandle = library.handle(
        EngineImplementation.Scale.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT),
    )
    private val copyHandle = library.handle(
        EngineImplementation.Copy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val swapHandle = library.handle(
        EngineImplementation.Swap.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val rotHandle = library.handle(
        EngineImplementation.Rot.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE),
    )
    private val gemvHandle = library.handle(
        EngineImplementation.Gemv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symvHandle = library.handle(
        EngineImplementation.Symv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val gerHandle = library.handle(
        EngineImplementation.Ger.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val syrHandle = library.handle(
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
    private val syr2Handle = library.handle(
        EngineImplementation.Syr2.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val trsvHandle = library.handle(EngineImplementation.Trsv.entryPoint, TRIANGULAR_VECTOR)
    private val trmvHandle = library.handle(EngineImplementation.Trmv.entryPoint, TRIANGULAR_VECTOR)
    private val gemmHandle = library.handle(
        EngineImplementation.Gemm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symmHandle = library.handle(
        EngineImplementation.Symm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syrkHandle = library.handle(
        EngineImplementation.Syrk.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syr2kHandle = library.handle(
        EngineImplementation.Syr2k.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val trmmHandle = library.handle(EngineImplementation.Trmm.entryPoint, TRIANGULAR_MATRIX)
    private val trsmHandle = library.handle(EngineImplementation.Trsm.entryPoint, TRIANGULAR_MATRIX)
    private val gemmtOrNull = library.handleOrNull(
        EngineImplementation.Gemmt.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )

    private companion object {
        /** Named in every route because every JVM call pays it. */
        const val TRANSFER = "native buffer transfer"

        val TRIANGULAR_VECTOR: FunctionDescriptor = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        )
        val TRIANGULAR_MATRIX: FunctionDescriptor = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        )
    }
}
