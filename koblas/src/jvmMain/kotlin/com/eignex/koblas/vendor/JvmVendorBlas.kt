@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dense.MatrixStructure
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
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

    // Level 2. One matrix operand, so the call runs under that matrix's own Cblas.COL_MAJOR.

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        requireGemvOperands(a, transposeA, x, y)
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
        requireSymvOperands(a, structure, x, y)
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
        requireGerOperands(x, y, a)
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
        requireSyrOperands(a, structure, "syr", x)
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
        requireSyr2Operands(a, structure, "syr2", x, y)
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
        requireTriangularVectorOperands(a, structure, x, what)
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
        requireGemmOperands(a, transposeA, b, transposeB, c)
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
        requireSymmOperands(a, structure, b, c, rightSide)
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
        requireSyrkOperands(a, transposeA, c, structure)
        val depth = if (transposeA) a.rows else a.cols
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
        requireSyr2kOperands(a, b, transposeA, c, structure)
        val depth = if (transposeA) a.rows else a.cols
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
        requireTriangularMatrixOperands(a, structure, b, rightSide, what)
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
        requireGemmtOperands(a, transposeA, b, transposeB, c, structure)
        val depth = if (transposeA) a.rows else a.cols
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
