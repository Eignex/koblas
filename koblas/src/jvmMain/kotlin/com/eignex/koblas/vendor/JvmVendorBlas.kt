@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
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
    private val suppressed: Set<VendorOperation> = emptySet(),
) : VendorBlas {
    override val vendor: Vendor get() = library.vendor
    override val libraryPath: String get() = library.resolvedFile
    override val version: String get() = library.version
    override val threadEvidence: ThreadEvidence get() = library.threadEvidence

    override val directlyImplemented: Set<VendorOperation> = VendorOperation.entries
        .filterTo(LinkedHashSet()) { it !in suppressed && library.exports(it.entryPoint) }

    override fun routeOf(operation: VendorOperation, matrices: List<MatrixWindow>): CallRoute = routeFor(
        operation = operation,
        vendor = vendor,
        exported = operation in directlyImplemented,
        matrices = matrices,
        transfer = TRANSFER,
    )

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    override fun dot(x: VectorWindow, y: VectorWindow): Double {
        requireSameLength(x, y, "dot")
        if (x.size == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            dotHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment) as Double
        }
    }

    override fun nrm2(x: VectorWindow): Double {
        if (x.size == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            nrm2Handle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Double
        }
    }

    override fun asum(x: VectorWindow): Double {
        if (x.size == 0) return 0.0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            asumHandle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Double
        }
    }

    override fun iamax(x: VectorWindow): Int {
        if (x.size == 0) return 0
        return Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val found = iamaxHandle.invokeExact(x.size, nx.segment, abs(nx.increment)) as Int
            if (x.stride >= 0) found else x.size - 1 - found
        }
    }

    override fun axpy(alpha: Double, x: VectorWindow, y: VectorWindow) {
        requireSameLength(x, y, "axpy")
        if (x.size == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            axpyHandle.invokeExact(x.size, alpha, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun scal(alpha: Double, x: VectorWindow) {
        if (x.size == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            scalHandle.invokeExact(x.size, alpha, nx.segment, nx.increment)
            nx.writeBack()
        }
    }

    override fun copy(x: VectorWindow, y: VectorWindow) {
        requireSameLength(x, y, "copy")
        if (x.size == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            copyHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            ny.writeBack()
        }
    }

    override fun swap(x: VectorWindow, y: VectorWindow) {
        requireSameLength(x, y, "swap")
        if (x.size == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            swapHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment)
            nx.writeBack()
            ny.writeBack()
        }
    }

    override fun rot(x: VectorWindow, y: VectorWindow, c: Double, s: Double) {
        requireSameLength(x, y, "rot")
        if (x.size == 0) return
        Arena.ofConfined().use { arena ->
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            rotHandle.invokeExact(x.size, nx.segment, nx.increment, ny.segment, ny.increment, c, s)
            nx.writeBack()
            ny.writeBack()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own layout.

    override fun gemv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow) {
        require(x.size == a.columns && y.size == a.rows) { "gemv: operand sizes do not match the matrix" }
        if (a.rows == 0 || a.columns == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            gemvHandle.invokeExact(
                layout, transposeFor(addressing, layout), a.rows, a.columns, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun symv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow) {
        requireStructured(a, "symv")
        require(x.size == a.columns && y.size == a.rows) { "symv: operand sizes do not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            symvHandle.invokeExact(
                layout, uploFor(a, addressing, layout), a.rows, alpha,
                na.segment, na.leadingDimension, nx.segment, nx.increment, beta, ny.segment, ny.increment,
            )
            ny.writeBack()
        }
    }

    override fun ger(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow) {
        require(x.size == a.rows && y.size == a.columns) { "ger: operand sizes do not match the matrix" }
        if (a.rows == 0 || a.columns == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            gerHandle.invokeExact(
                layout, a.rows, a.columns, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun syr(alpha: Double, x: VectorWindow, a: MatrixWindow) {
        requireStructured(a, "syr")
        require(x.size == a.rows) { "syr: operand size does not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            syrHandle.invokeExact(
                layout,
                uploFor(a, addressing, layout),
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

    override fun syr2(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow) {
        requireStructured(a, "syr2")
        require(x.size == a.rows && y.size == a.rows) { "syr2: operand sizes do not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            val ny = arena.stage(y)
            syr2Handle.invokeExact(
                layout, uploFor(a, addressing, layout), a.rows, alpha,
                nx.segment, nx.increment, ny.segment, ny.increment, na.segment, na.leadingDimension,
            )
            na.writeBack()
        }
    }

    override fun trsv(a: MatrixWindow, x: VectorWindow) = triangularVector(a, x, trsvHandle, "trsv")

    override fun trmv(a: MatrixWindow, x: VectorWindow) = triangularVector(a, x, trmvHandle, "trmv")

    private fun triangularVector(a: MatrixWindow, x: VectorWindow, handle: MethodHandle, what: String) {
        requireTriangular(a, what)
        require(x.size == a.rows) { "$what: operand size does not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing)
            val nx = arena.stage(x)
            handle.invokeExact(
                layout, uploFor(a, addressing, layout), transposeFor(addressing, layout), diagFor(a),
                a.rows, na.segment, na.leadingDimension, nx.segment, nx.increment,
            )
            nx.writeBack()
        }
    }

    // Level 3. The layout is settled across every matrix operand before any of them is staged.

    override fun gemm(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        require(a.columns == b.rows && c.rows == a.rows && c.columns == b.columns) { "gemm: shapes do not conform" }
        if (c.rows == 0 || c.columns == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Gemm, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nb = arena.stage(b, addressing[1])
            val nc = arena.stage(c, addressing[2])
            gemmHandle.invokeExact(
                layout, transposeFor(addressing[0], layout), transposeFor(addressing[1], layout),
                c.rows, c.columns, a.columns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun symm(
        alpha: Double,
        a: MatrixWindow,
        b: MatrixWindow,
        beta: Double,
        c: MatrixWindow,
        rightSide: Boolean,
    ) {
        requireStructured(a, "symm")
        require(c.rows == b.rows && c.columns == b.columns) { "symm: shapes do not conform" }
        if (c.rows == 0 || c.columns == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Symm, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nb = arena.stage(b, addressing[1])
            val nc = arena.stage(c, addressing[2])
            symmHandle.invokeExact(
                layout, sideFor(rightSide), uploFor(a, addressing[0], layout), c.rows, c.columns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun syrk(alpha: Double, a: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "syrk")
        require(c.rows == a.rows) { "syrk: shapes do not conform" }
        if (c.rows == 0) return
        val operands = listOf(a, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Syrk, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nc = arena.stage(c, addressing[1])
            syrkHandle.invokeExact(
                layout, uploFor(c, addressing[1], layout), transposeFor(addressing[0], layout),
                c.rows, a.columns, alpha, na.segment, na.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun syr2k(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "syr2k")
        require(c.rows == a.rows && a.rows == b.rows && a.columns == b.columns) { "syr2k: shapes do not conform" }
        if (c.rows == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Syr2k, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nb = arena.stage(b, addressing[1])
            val nc = arena.stage(c, addressing[2])
            syr2kHandle.invokeExact(
                layout, uploFor(c, addressing[2], layout), transposeFor(addressing[0], layout),
                c.rows, a.columns, alpha, na.segment, na.leadingDimension,
                nb.segment, nb.leadingDimension, beta, nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    override fun trmm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean) =
        triangularMatrix(alpha, a, b, rightSide, trmmHandle, VendorOperation.Trmm, "trmm")

    override fun trsm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean) =
        triangularMatrix(alpha, a, b, rightSide, trsmHandle, VendorOperation.Trsm, "trsm")

    private fun triangularMatrix(
        alpha: Double,
        a: MatrixWindow,
        b: MatrixWindow,
        rightSide: Boolean,
        handle: MethodHandle,
        operation: VendorOperation,
        what: String,
    ) {
        requireTriangular(a, what)
        require(if (rightSide) a.rows == b.columns else a.rows == b.rows) { "$what: shapes do not conform" }
        if (b.rows == 0 || b.columns == 0) return
        val operands = listOf(a, b)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(operation, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nb = arena.stage(b, addressing[1])
            handle.invokeExact(
                layout, sideFor(rightSide), uploFor(a, addressing[0], layout),
                transposeFor(addressing[0], layout), diagFor(a), b.rows, b.columns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension,
            )
            nb.writeBack()
        }
    }

    /**
     * Direct where the vendor exports `cblas_dgemmt`, and otherwise a full [gemm] into scratch followed by a
     * copy of the selected triangle.
     *
     * The composed path keeps the contract the direct one has: the opposite triangle of [c] is neither read
     * nor written, and a zero [beta] does not read the destination. It costs a full product either way, which
     * is the reason the route says which path ran rather than reporting both as `gemmt`.
     */
    override fun gemmt(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "gemmt")
        require(a.columns == b.rows && c.rows == a.rows && c.columns == b.columns) { "gemmt: shapes do not conform" }
        if (c.rows == 0 || c.columns == 0) return
        if (VendorOperation.Gemmt !in directlyImplemented) return composedGemmt(alpha, a, b, beta, c)
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Gemmt, operands)
        Arena.ofConfined().use { arena ->
            val na = arena.stage(a, addressing[0])
            val nb = arena.stage(b, addressing[1])
            val nc = arena.stage(c, addressing[2])
            gemmtHandle().invokeExact(
                layout, uploFor(c, addressing[2], layout), transposeFor(addressing[0], layout),
                transposeFor(addressing[1], layout), c.rows, a.columns, alpha,
                na.segment, na.leadingDimension, nb.segment, nb.leadingDimension, beta,
                nc.segment, nc.leadingDimension,
            )
            nc.writeBack()
        }
    }

    private fun composedGemmt(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        val order = c.rows
        val product = DoubleArray(order * order)
        gemm(alpha, a, b, 0.0, MatrixWindow(product, order, order))
        val lower = c.structure == MatrixStructure.SymmetricLower ||
            c.structure == MatrixStructure.TriangularLower
        for (column in 0 until order) {
            val from = if (lower) column else 0
            val until = if (lower) order else column + 1
            for (row in from until until) {
                val index = c.index(row, column)
                val updated = if (beta == 0.0) 0.0 else beta * c.data[index]
                c.data[index] = updated + product[row + column * order]
            }
        }
    }

    private fun gemmtHandle(): MethodHandle = checkNotNull(gemmtOrNull) { "gemmt is not exported" }

    private val dotHandle = library.handle(
        VendorOperation.Dot.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val nrm2Handle = library.handle(
        VendorOperation.Nrm2.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val asumHandle = library.handle(
        VendorOperation.Asum.entryPoint,
        FunctionDescriptor.of(JAVA_DOUBLE, JAVA_INT, ADDRESS, JAVA_INT),
    )

    /**
     * `CBLAS_INDEX` is `size_t` in the reference header and in several vendors, and `int` in others. The low
     * 32 bits agree for every index a 32-bit BLAS integer can address, so binding it as `int` reads the same
     * value either way and avoids depending on which one this library used.
     */
    private val iamaxHandle = library.handle(
        VendorOperation.Iamax.entryPoint,
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val axpyHandle = library.handle(
        VendorOperation.Axpy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val scalHandle = library.handle(
        VendorOperation.Scal.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT),
    )
    private val copyHandle = library.handle(
        VendorOperation.Copy.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val swapHandle = library.handle(
        VendorOperation.Swap.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT),
    )
    private val rotHandle = library.handle(
        VendorOperation.Rot.entryPoint,
        FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, JAVA_DOUBLE),
    )
    private val gemvHandle = library.handle(
        VendorOperation.Gemv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symvHandle = library.handle(
        VendorOperation.Symv.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val gerHandle = library.handle(
        VendorOperation.Ger.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val syrHandle = library.handle(
        VendorOperation.Syr.entryPoint,
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
        VendorOperation.Syr2.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT,
        ),
    )
    private val trsvHandle = library.handle(VendorOperation.Trsv.entryPoint, TRIANGULAR_VECTOR)
    private val trmvHandle = library.handle(VendorOperation.Trmv.entryPoint, TRIANGULAR_VECTOR)
    private val gemmHandle = library.handle(
        VendorOperation.Gemm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val symmHandle = library.handle(
        VendorOperation.Symm.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syrkHandle = library.handle(
        VendorOperation.Syrk.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val syr2kHandle = library.handle(
        VendorOperation.Syr2k.entryPoint,
        FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_DOUBLE,
            ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT,
        ),
    )
    private val trmmHandle = library.handle(VendorOperation.Trmm.entryPoint, TRIANGULAR_MATRIX)
    private val trsmHandle = library.handle(VendorOperation.Trsm.entryPoint, TRIANGULAR_MATRIX)
    private val gemmtOrNull = library.handleOrNull(
        VendorOperation.Gemmt.entryPoint,
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

        fun requireSameLength(x: VectorWindow, y: VectorWindow, what: String) {
            require(x.size == y.size) { "$what: vector sizes differ" }
        }

        fun requireStructured(a: MatrixWindow, what: String) {
            require(a.structure != MatrixStructure.General) { "$what requires a stored triangle" }
            require(a.rows == a.columns) { "$what requires a square matrix" }
        }

        fun requireTriangular(a: MatrixWindow, what: String) {
            val triangular = a.structure == MatrixStructure.TriangularLower ||
                a.structure == MatrixStructure.TriangularUpper ||
                a.structure == MatrixStructure.UnitLower ||
                a.structure == MatrixStructure.UnitUpper
            require(triangular) { "$what requires a triangular matrix" }
            require(a.rows == a.columns) { "$what requires a square matrix" }
        }
    }
}

/** Opens the preferred available vendor for this host, or null when none is installed. */
public actual fun openVendorBlas(only: Vendor?): VendorBlas? {
    val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
    return candidates.firstNotNullOfOrNull { vendor ->
        JvmVendorLibrary.open(vendor)?.let(::JvmVendorBlas)
    }
}
