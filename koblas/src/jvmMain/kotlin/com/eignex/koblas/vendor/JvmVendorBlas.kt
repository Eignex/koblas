@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.ModifiedGivens
import com.eignex.koblas.dense.MatrixStructure
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle
import kotlin.math.abs

/**
 * A vendor BLAS reached through `java.lang.foreign`.
 *
 * Handles are bound once at construction. Each call opens a confined arena, copies its operands in, makes one
 * non-critical downcall, copies its outputs back, and closes the arena on the way out including on an
 * exception. Nothing is cached between calls and no segment escapes one, so instances are safe to share and
 * concurrent calls on one instance do not interfere.
 */
internal class JvmVendorBlas(
    private val library: JvmVendorLibrary,
    /**
     * Operations to treat as absent however the library is built.
     *
     * Whether a vendor exports `cblas_dgemmt` is a property of the install, so on a host whose library has it
     * there is otherwise no way to reach the composed path at all. Suppressing it is how a test exercises the
     * branch a vendor without it would take. Nothing in production passes anything here.
     */
    private val suppressed: Set<BlasOperation> = emptySet(),
) : Blas {
    override val vendor: Vendor get() = library.vendor
    override val libraryPath: String get() = library.resolvedFile
    override val version: String get() = library.version
    override val threadEvidence: ThreadEvidence get() = library.threadEvidence

    override val directlyImplemented: Set<BlasOperation> = BlasOperation.entries
        .filterTo(LinkedHashSet()) { it !in suppressed && library.exports(it.entryPoint) }

    override fun routeOf(
        operation: BlasOperation,
        matrices: List<DenseMatrix>,
        vectors: List<DenseVector>,
    ): CallRoute = routeFor(
        operation = operation,
        vendor = vendor,
        exported = operation in directlyImplemented,
        matrices = matrices,
        vectors = vectors,
        transfer = TRANSFER,
    )

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    override fun dot(x: DenseVector, y: DenseVector): Double {
        requireSameLength(x, y, "dot")
        if (noWorkReason(emptyList(), listOf(x)) != null) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            dotHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(x: DenseVector): Double {
        if (noWorkReason(emptyList(), listOf(x)) != null) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            nrm2Handle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Double
        }
    }

    override fun asum(x: DenseVector): Double {
        if (noWorkReason(emptyList(), listOf(x)) != null) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            asumHandle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Double
        }
    }

    override fun iamax(x: DenseVector): Int {
        if (noWorkReason(emptyList(), listOf(x)) != null) return 0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val found = iamaxHandle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Int
            if (x.stride >= 0) found else x.size - 1 - found
        }
    }

    override fun axpy(alpha: Double, x: DenseVector, y: DenseVector) {
        requireSameLength(x, y, "axpy")
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            axpyHandle.invokeExact(x.size, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scal(alpha: Double, x: DenseVector) {
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            scalHandle.invokeExact(x.size, alpha, nx.segment, abs(nx.increment))
            nx.writeBack()
        }
    }

    override fun copy(x: DenseVector, y: DenseVector) {
        requireSameLength(x, y, "copy")
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            copyHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun swap(x: DenseVector, y: DenseVector) {
        requireSameLength(x, y, "swap")
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            swapHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            nx.writeBack()
            ny.writeBack()
        }
    }

    override fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
        requireSameLength(x, y, "rot")
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            rotHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    override fun rotmg(d1: Double, d2: Double, x1: Double, y1: Double): ModifiedGivens {
        val handle = rotmgHandle
        return Arena.ofConfined().use { arena ->
            // d1, d2 and x1 are read and written in place, so each goes over as its own cell.
            val pd1 = arena.allocateFrom(JAVA_DOUBLE, d1)
            val pd2 = arena.allocateFrom(JAVA_DOUBLE, d2)
            val px1 = arena.allocateFrom(JAVA_DOUBLE, x1)
            val param = arena.allocate(JAVA_DOUBLE, PARAM_ENTRIES.toLong())
            handle.invokeExact(pd1, pd2, px1, y1, param)
            param.readModifiedGivens(
                pd1.get(JAVA_DOUBLE, 0),
                pd2.get(JAVA_DOUBLE, 0),
                px1.get(JAVA_DOUBLE, 0),
            )
        }
    }

    override fun rotm(x: DenseVector, y: DenseVector, transformation: ModifiedGivens) {
        requireSameLength(x, y, "rotm")
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        val handle = rotmHandle
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            val param = arena.allocate(JAVA_DOUBLE, PARAM_ENTRIES.toLong())
            param.writeModifiedGivens(transformation)
            handle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment, param)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        // The flag decides which dimension each operand answers to, so reading it is part of the check.
        val expectedX = if (transposeA) a.rows else a.cols
        val expectedY = if (transposeA) a.cols else a.rows
        require(x.size == expectedX && y.size == expectedY) { "gemv: operand sizes do not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            gemvHandle.invokeExact(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        requireStructured(a, structure, "symv")
        require(x.size == a.cols && y.size == a.rows) { "symv: operand sizes do not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            symvHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix) {
        require(x.size == a.rows && y.size == a.cols) { "ger: operand sizes do not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
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

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(a, structure, "syr")
        require(x.size == a.rows) { "syr: operand size does not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
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

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(a, structure, "syr2")
        require(x.size == a.rows && y.size == a.rows) { "syr2: operand sizes do not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            syr2Handle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun trsv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, trsvHandle, "trsv")

    override fun trmv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, trmvHandle, "trmv")

    @Suppress("LongParameterList") // the shared triangular vector signature plus its handle
    private fun triangularVector(
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
        handle: MethodHandle,
        what: String,
    ) {
        requireTriangular(a, structure, what)
        require(x.size == a.rows) { "$what: operand size does not match the matrix" }
        if (noWorkReason(listOf(a), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
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
    override fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
    ) {
        val depth = if (transposeA) a.rows else a.cols
        require(c.rows == (if (transposeA) a.cols else a.rows)) { "gemm: shapes do not conform" }
        require(c.cols == (if (transposeB) b.rows else b.cols)) { "gemm: shapes do not conform" }
        require(depth == (if (transposeB) b.cols else b.rows)) { "gemm: shapes do not conform" }
        if (noWorkReason(listOf(c), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c)
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
    override fun symm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        b: DenseMatrix,
        beta: Double,
        c: DenseMatrix,
        rightSide: Boolean,
    ) {
        requireStructured(a, structure, "symm")
        require(c.rows == b.rows && c.cols == b.cols) { "symm: shapes do not conform" }
        require(a.rows == if (rightSide) c.cols else c.rows) { "symm: the symmetric operand has the wrong order" }
        if (noWorkReason(listOf(c), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c)
            symmHandle.invokeExact(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature
    override fun syrk(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(c, structure, "syrk")
        val depth = if (transposeA) a.rows else a.cols
        require(c.rows == (if (transposeA) a.cols else a.rows)) { "syrk: shapes do not conform" }
        if (noWorkReason(listOf(c), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nc = arena.stage(c)
            syrkHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyr2k signature
    override fun syr2k(
        alpha: Double,
        a: DenseMatrix,
        b: DenseMatrix,
        transposeA: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(c, structure, "syr2k")
        val depth = if (transposeA) a.rows else a.cols
        require(a.rows == b.rows && a.cols == b.cols) { "syr2k: shapes do not conform" }
        require(c.rows == (if (transposeA) a.cols else a.rows)) { "syr2k: shapes do not conform" }
        if (noWorkReason(listOf(c), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c)
            syr2kHandle.invokeExact(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    @Suppress("LongParameterList") // the BLAS dtrmm signature
    override fun trmm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, trmmHandle, "trmm")

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, trsmHandle, "trsm")

    @Suppress("LongParameterList") // the shared triangular matrix signature plus its handle
    private fun triangularMatrix(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
        handle: MethodHandle,
        what: String,
    ) {
        requireTriangular(a, structure, what)
        require(if (rightSide) a.rows == b.cols else a.rows == b.rows) { "$what: shapes do not conform" }
        if (noWorkReason(listOf(b), emptyList()) != null) return
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
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
    override fun gemmt(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        structure: MatrixStructure,
    ) {
        requireStructured(c, structure, "gemmt")
        val depth = if (transposeA) a.rows else a.cols
        require(c.rows == (if (transposeA) a.cols else a.rows)) { "gemmt: shapes do not conform" }
        require(c.cols == (if (transposeB) b.rows else b.cols)) { "gemmt: shapes do not conform" }
        if (noWorkReason(listOf(c), emptyList()) != null) return
        if (BlasOperation.Gemmt !in directlyImplemented) {
            return composeGemmt(alpha, a, transposeA, b, transposeB, beta, c, structure)
        }
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a)
            val nb = arena.stage(b)
            val nc = arena.stage(c)
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
        BlasOperation.Dot.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val nrm2Handle = library.handle(
        BlasOperation.Nrm2.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val asumHandle = library.handle(
        BlasOperation.Asum.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )

    /**
     * `CBLAS_INDEX` is `size_t` in the reference header and in several vendors, and `int` in others. The low
     * 32 bits agree for every index a 32-bit BLAS integer can address, so binding it as `int` reads the same
     * value either way and avoids depending on which one this library used.
     */
    private val iamaxHandle = library.handle(
        BlasOperation.Iamax.entryPoint,
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val axpyHandle = library.handle(
        BlasOperation.Axpy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val scalHandle = library.handle(
        BlasOperation.Scal.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT),
    )
    private val copyHandle = library.handle(
        BlasOperation.Copy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val swapHandle = library.handle(
        BlasOperation.Swap.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val rotHandle = library.handle(
        BlasOperation.Rot.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE),
    )
    private val rotmgHandle = library.handle(
        BlasOperation.Rotmg.entryPoint,
        FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, JAVA_DOUBLE, ADDRESS),
    )
    private val rotmHandle = library.handle(
        BlasOperation.Rotm.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS),
    )
    private val gemvHandle = library.handle(
        BlasOperation.Gemv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symvHandle = library.handle(
        BlasOperation.Symv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val gerHandle = library.handle(
        BlasOperation.Ger.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val syrHandle = library.handle(
        BlasOperation.Syr.entryPoint,
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
        BlasOperation.Syr2.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val trsvHandle = library.handle(BlasOperation.Trsv.entryPoint, TRIANGULAR_VECTOR)
    private val trmvHandle = library.handle(BlasOperation.Trmv.entryPoint, TRIANGULAR_VECTOR)
    private val gemmHandle = library.handle(
        BlasOperation.Gemm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symmHandle = library.handle(
        BlasOperation.Symm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syrkHandle = library.handle(
        BlasOperation.Syrk.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syr2kHandle = library.handle(
        BlasOperation.Syr2k.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val trmmHandle = library.handle(BlasOperation.Trmm.entryPoint, TRIANGULAR_MATRIX)
    private val trsmHandle = library.handle(BlasOperation.Trsm.entryPoint, TRIANGULAR_MATRIX)
    private val gemmtOrNull = library.handleOrNull(
        BlasOperation.Gemmt.entryPoint,
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

/** Opens the preferred available vendor for this host, or null when none is installed. */
public actual fun openBlas(only: Vendor?): Blas? {
    val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
    return candidates.firstNotNullOfOrNull { vendor ->
        JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
    }
}

/**
 * Entries in the BLAS modified-Givens parameter array.
 *
 * The array is `[flag, h11, h21, h12, h22]`, which is the 2x2 matrix in column-major order rather than the
 * reading order of its name. Writing it row-major transposes the rotation, and because the result is still a
 * plausible rotation nothing downstream would report it, so the order is named here once and used from both
 * directions.
 */
private const val PARAM_ENTRIES = 5

private fun MemorySegment.writeModifiedGivens(transformation: ModifiedGivens) {
    setAtIndex(JAVA_DOUBLE, 0, transformation.flag)
    setAtIndex(JAVA_DOUBLE, 1, transformation.h11)
    setAtIndex(JAVA_DOUBLE, 2, transformation.h21)
    setAtIndex(JAVA_DOUBLE, 3, transformation.h12)
    setAtIndex(JAVA_DOUBLE, 4, transformation.h22)
}

private fun MemorySegment.readModifiedGivens(d1: Double, d2: Double, x1: Double): ModifiedGivens = ModifiedGivens(
    d1 = d1,
    d2 = d2,
    x1 = x1,
    flag = getAtIndex(JAVA_DOUBLE, 0),
    h11 = getAtIndex(JAVA_DOUBLE, 1),
    h21 = getAtIndex(JAVA_DOUBLE, 2),
    h12 = getAtIndex(JAVA_DOUBLE, 3),
    h22 = getAtIndex(JAVA_DOUBLE, 4),
)
