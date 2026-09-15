@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.convert
import kotlinx.cinterop.get
import kotlinx.cinterop.invoke
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import platform.posix.RTLD_NOW
import platform.posix.dlclose
import platform.posix.dlopen
import platform.posix.dlsym
import platform.posix.getenv
import platform.posix.readlink
import platform.posix.setenv
import kotlin.math.abs

private typealias Ptr = CPointer<DoubleVar>
private typealias Bytes = CPointer<ByteVar>
private typealias DotFn = CFunction<(Int, Ptr?, Int, Ptr?, Int) -> Double>
private typealias ReduceFn = CFunction<(Int, Ptr?, Int) -> Double>
private typealias IndexFn = CFunction<(Int, Ptr?, Int) -> Int>
private typealias AxpyFn = CFunction<(Int, Double, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias ScalFn = CFunction<(Int, Double, Ptr?, Int) -> Unit>
private typealias CopyFn = CFunction<(Int, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias RotFn = CFunction<(Int, Ptr?, Int, Ptr?, Int, Double, Double) -> Unit>
private typealias GemvFn = CFunction<(Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias SymvFn = CFunction<(Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias GerFn = CFunction<(Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias SyrFn = CFunction<(Int, Int, Int, Double, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias Syr2Fn = CFunction<(Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias TriangularVectorFn = CFunction<(Int, Int, Int, Int, Int, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias GemmFn =
    CFunction<(Int, Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias SymmFn = CFunction<(Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias SyrkFn = CFunction<(Int, Int, Int, Int, Int, Double, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias Syr2kFn =
    CFunction<(Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias TriangularMatrixFn =
    CFunction<(Int, Int, Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias VersionFn = CFunction<(Bytes?, Int) -> Unit>
private typealias StringFn = CFunction<() -> Bytes?>
private typealias SetLayerFn = CFunction<(Int) -> Int>
private typealias SetThreadsFn = CFunction<(Int) -> Unit>
private typealias GetThreadsFn = CFunction<() -> Int>
private typealias DlAddrFn = CFunction<(COpaquePointer?, CPointer<ByteVar>?) -> Int>

/**
 * A vendor BLAS reached through `dlopen` and typed function pointers.
 *
 * The library handle is opened once and held for the process, and every symbol is resolved out of that handle
 * rather than out of the global namespace. The distinction matters on a host with more than one BLAS loaded:
 * a global lookup for `cblas_dgemm` returns whichever the loader bound first, so a run labelled with one
 * vendor could be executing another.
 *
 * Operands are pinned rather than copied, so a call reaches the caller's own storage directly and the route of
 * a directly addressed call names no adapter at all.
 */
internal class NativeVendorBlas private constructor(
    override val vendor: Vendor,
    private val candidate: String,
    private val handle: COpaquePointer,
) : VendorBlas {
    /**
     * The file the key symbol actually came from, asked of the dynamic loader rather than assumed.
     *
     * The candidate that opened is not the same thing: a bare soname matches a library already loaded into the
     * process by some earlier absolute-path open, so two runs can record different strings for one file
     * depending only on what ran first.
     */
    override val libraryPath: String by lazy { symbolOwner() ?: candidate }

    private fun symbolOwner(): String? = memScoped {
        val address = dlsym(handle, vendor.keySymbol) ?: return@memScoped null
        // Reached through the library's own dependency chain, which includes libc: dladdr is a GNU extension
        // that Kotlin/Native does not surface, and this needs no cinterop of its own.
        val dladdr = dlsym(handle, "dladdr")?.reinterpret<DlAddrFn>() ?: return@memScoped null
        val info = allocArray<ByteVar>(DL_INFO_BYTES)
        if (dladdr(address, info) == 0) return@memScoped null
        val name = info.reinterpret<CPointerVar<ByteVar>>()[0] ?: return@memScoped null
        name.toKString().ifEmpty { null }
    }

    override val version: String by lazy { readVersion() }

    /**
     * Pins the library to one compute thread per call, before any arithmetic reaches it.
     *
     * Every one of these libraries is multithreaded by default, so this is what makes a call single-threaded
     * rather than a preference expressed about it. It runs once, at load, and there is no way to reach it
     * afterwards: the thread count is an invariant of the binding and not a setting it carries.
     *
     * oneMKL takes two steps. Its dispatcher resolves a threading layer on first use and defaults to the
     * Intel-threaded one, which needs an OpenMP runtime a plain oneMKL install does not ship; on a host without
     * `libiomp5` the first call dies with an undefined `omp_get_num_procs` instead of computing. Naming the
     * sequential layer both makes the library usable and removes its workers. The thread count and dynamic
     * expansion are then fixed as well, so a build that resolves some other layer cannot grow workers back.
     *
     * A library that ignores all of this is caught by [confirmSingleThread] rather than trusted. That read-back
     * happens after the ABI probe, because the probe is itself arithmetic and must not be what resolves the
     * layer.
     */
    private fun enforceSingleThread() {
        // Accelerate has no thread-count entry point; see ACCELERATE_THREAD_LIMIT. Overwrite is on so a value
        // inherited from the launching shell cannot weaken the invariant.
        if (vendor == Vendor.Accelerate) setenv(ACCELERATE_THREAD_LIMIT, "1", 1)
        dlsym(handle, "MKL_Set_Threading_Layer")?.reinterpret<SetLayerFn>()?.invoke(MKL_SEQUENTIAL)
        dlsym(handle, "MKL_Set_Dynamic")?.reinterpret<SetThreadsFn>()?.invoke(0)
        for (symbol in listOf("MKL_Set_Num_Threads", "openblas_set_num_threads", "bli_thread_set_num_threads")) {
            dlsym(handle, symbol)?.reinterpret<SetThreadsFn>()?.invoke(1)
        }
    }

    /** Set once at load, after [enforceSingleThread], and never again. */
    override var threadEvidence: ThreadEvidence = ThreadEvidence.Unconfirmed
        private set

    /**
     * Holds the library to one compute thread and records whether that could be read back.
     *
     * Returns false for a library that still reports more than one thread, which is how such a library is kept
     * out of arithmetic entirely rather than becoming a silently multithreaded arm. One that exposes no way to
     * ask is accepted on the strength of the request, which is all there is to go on; [Vendor.Accelerate] is
     * the case that matters, since it carries no thread-count entry point of its own.
     */
    private fun confirmSingleThread(): Boolean {
        for (symbol in listOf("MKL_Get_Max_Threads", "openblas_get_num_threads", "bli_thread_get_num_threads")) {
            val fn = dlsym(handle, symbol)?.reinterpret<GetThreadsFn>() ?: continue
            if (fn() != 1) return false
            threadEvidence = ThreadEvidence.Confirmed
            return true
        }
        threadEvidence = ThreadEvidence.Unconfirmed
        return true
    }

    /**
     * Whether the library matches the ABI Koblas binds and computes correctly through it.
     *
     * The integer width is read from what the build says about itself, because a call cannot distinguish it:
     * a 32-bit argument arrives in a register whose upper half is zeroed, so an ILP64 routine reading 64 bits
     * sees the same small value. Whether the library computes at all is settled by calling it with operands
     * whose exact answer is known.
     */
    private fun verifiedAbi(): Boolean {
        if (declaresWideIntegers(version)) return false
        if (!declaredIntegerWidthMatches()) return false
        return probeDot() && probeGemm()
    }

    /**
     * BLIS answers the integer-width question directly; see [BLIS_INTEGER_WIDTH].
     *
     * It takes no arguments and returns an `int`, which is the shape [GetThreadsFn] already names.
     */
    private fun declaredIntegerWidthMatches(): Boolean {
        val fn = dlsym(handle, BLIS_INTEGER_WIDTH)?.reinterpret<GetThreadsFn>() ?: return true
        return fn() == LP64_INTEGER_BITS
    }

    private fun probeDot(): Boolean {
        val fn = dlsym(handle, VendorOperation.Dot.entryPoint)?.reinterpret<DotFn>() ?: return false
        val pins = Pins()
        try {
            val x = AbiProbe.x
            val y = AbiProbe.y
            return fn(x.size, pins.pointer(x, 0), 1, pins.pointer(y, 0), 1) == AbiProbe.DOT
        } finally {
            pins.release()
        }
    }

    private fun probeGemm(): Boolean {
        val fn = dlsym(handle, VendorOperation.Gemm.entryPoint)?.reinterpret<GemmFn>() ?: return false
        val pins = Pins()
        try {
            val a = AbiProbe.identity
            val b = AbiProbe.operand
            val c = DoubleArray(b.size)
            fn(
                Cblas.COL_MAJOR, Cblas.NO_TRANS, Cblas.NO_TRANS, 2, 2, 2, 1.0,
                pins.pointer(a, 0), 2, pins.pointer(b, 0), 2, 0.0, pins.pointer(c, 0), 2,
            )
            return c.indices.all { c[it] == b[it] }
        } finally {
            pins.release()
        }
    }

    /**
     * Every entry point resolved once, at construction.
     *
     * `dlsym` is a lock and a hash lookup through the library's dependency chain, and doing it per call would
     * put that on the path of every Level 1 operation, where it is a large fraction of the work being timed.
     * The handle is held for the process, so what it resolves to cannot change underneath this.
     */
    private val entryPoints: Array<COpaquePointer?> =
        Array(VendorOperation.entries.size) { dlsym(handle, VendorOperation.entries[it].entryPoint) }

    override val directlyImplemented: Set<VendorOperation> =
        VendorOperation.entries.filterTo(LinkedHashSet()) { entryPoints[it.ordinal] != null }

    override fun routeOf(operation: VendorOperation, matrices: List<MatrixWindow>): CallRoute = routeFor(
        operation = operation,
        vendor = vendor,
        exported = operation in directlyImplemented,
        matrices = matrices,
        transfer = null,
    )

    private fun symbol(operation: VendorOperation): COpaquePointer = checkNotNull(entryPoints[operation.ordinal]) {
        "${vendor.vendorName} at $libraryPath does not export ${operation.entryPoint}"
    }

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    override fun dot(x: VectorWindow, y: VectorWindow): Double {
        requireSameLength(x, y, "dot")
        if (x.size == 0) return 0.0
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Dot).reinterpret<DotFn>()
            return fn(x.size, px.pointer, px.increment, py.pointer, py.increment)
        } finally {
            pins.release()
        }
    }

    override fun nrm2(x: VectorWindow): Double = reduce(x, VendorOperation.Nrm2)

    override fun asum(x: VectorWindow): Double = reduce(x, VendorOperation.Asum)

    private fun reduce(x: VectorWindow, operation: VendorOperation): Double {
        if (x.size == 0) return 0.0
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val fn = symbol(operation).reinterpret<ReduceFn>()
            return fn(x.size, px.pointer, abs(px.increment))
        } finally {
            pins.release()
        }
    }

    override fun iamax(x: VectorWindow): Int {
        if (x.size == 0) return 0
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val fn = symbol(VendorOperation.Iamax).reinterpret<IndexFn>()
            val found = fn(x.size, px.pointer, abs(px.increment))
            return if (x.stride >= 0) found else x.size - 1 - found
        } finally {
            pins.release()
        }
    }

    override fun axpy(alpha: Double, x: VectorWindow, y: VectorWindow) {
        requireSameLength(x, y, "axpy")
        if (x.size == 0) return
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Axpy).reinterpret<AxpyFn>()
            fn(x.size, alpha, px.pointer, px.increment, py.pointer, py.increment)
        } finally {
            pins.release()
        }
    }

    override fun scal(alpha: Double, x: VectorWindow) {
        if (x.size == 0) return
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val fn = symbol(VendorOperation.Scal).reinterpret<ScalFn>()
            fn(x.size, alpha, px.pointer, abs(px.increment))
        } finally {
            pins.release()
        }
    }

    override fun copy(x: VectorWindow, y: VectorWindow) = twoVector(x, y, VendorOperation.Copy, "copy")

    override fun swap(x: VectorWindow, y: VectorWindow) = twoVector(x, y, VendorOperation.Swap, "swap")

    private fun twoVector(x: VectorWindow, y: VectorWindow, operation: VendorOperation, what: String) {
        requireSameLength(x, y, what)
        if (x.size == 0) return
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(operation).reinterpret<CopyFn>()
            fn(x.size, px.pointer, px.increment, py.pointer, py.increment)
        } finally {
            pins.release()
        }
    }

    override fun rot(x: VectorWindow, y: VectorWindow, c: Double, s: Double) {
        requireSameLength(x, y, "rot")
        if (x.size == 0) return
        val pins = Pins()
        try {
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Rot).reinterpret<RotFn>()
            fn(x.size, px.pointer, px.increment, py.pointer, py.increment, c, s)
        } finally {
            pins.release()
        }
    }

    // Level 2. One matrix operand, so the call runs under that matrix's own layout.

    override fun gemv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow) {
        require(x.size == a.columns && y.size == a.rows) { "gemv: operand sizes do not match the matrix" }
        if (a.rows == 0 || a.columns == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Gemv).reinterpret<GemvFn>()
            fn(
                layout, transposeFor(addressing, layout), a.rows, a.columns, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
        } finally {
            pins.release()
        }
    }

    override fun symv(alpha: Double, a: MatrixWindow, x: VectorWindow, beta: Double, y: VectorWindow) {
        requireStructured(a, "symv")
        require(x.size == a.columns && y.size == a.rows) { "symv: operand sizes do not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Symv).reinterpret<SymvFn>()
            fn(
                layout, uploFor(a, addressing, layout), a.rows, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
        } finally {
            pins.release()
        }
    }

    override fun ger(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow) {
        require(x.size == a.rows && y.size == a.columns) { "ger: operand sizes do not match the matrix" }
        if (a.rows == 0 || a.columns == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Ger).reinterpret<GerFn>()
            fn(
                layout, a.rows, a.columns, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
            pa.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun syr(alpha: Double, x: VectorWindow, a: MatrixWindow) {
        requireStructured(a, "syr")
        require(x.size == a.rows) { "syr: operand size does not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val fn = symbol(VendorOperation.Syr).reinterpret<SyrFn>()
            fn(
                layout,
                uploFor(a, addressing, layout),
                a.rows,
                alpha,
                px.pointer,
                px.increment,
                pa.pointer,
                pa.leadingDimension,
            )
            pa.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun syr2(alpha: Double, x: VectorWindow, y: VectorWindow, a: MatrixWindow) {
        requireStructured(a, "syr2")
        require(x.size == a.rows && y.size == a.rows) { "syr2: operand sizes do not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(VendorOperation.Syr2).reinterpret<Syr2Fn>()
            fn(
                layout, uploFor(a, addressing, layout), a.rows, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
            pa.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun trsv(a: MatrixWindow, x: VectorWindow) = triangularVector(a, x, VendorOperation.Trsv, "trsv")

    override fun trmv(a: MatrixWindow, x: VectorWindow) = triangularVector(a, x, VendorOperation.Trmv, "trmv")

    private fun triangularVector(a: MatrixWindow, x: VectorWindow, operation: VendorOperation, what: String) {
        requireTriangular(a, what)
        require(x.size == a.rows) { "$what: operand size does not match the matrix" }
        if (a.rows == 0) return
        val layout = layoutOf(listOf(a))
        val addressing = addressingUnder(a, layout, absorbs = true)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing)
            val px = pins.stage(x)
            val fn = symbol(operation).reinterpret<TriangularVectorFn>()
            fn(
                layout, uploFor(a, addressing, layout), transposeFor(addressing, layout), diagFor(a),
                a.rows, pa.pointer, pa.leadingDimension, px.pointer, px.increment,
            )
        } finally {
            pins.release()
        }
    }

    // Level 3. The layout is settled across every matrix operand before any of them is staged.

    override fun gemm(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        require(a.columns == b.rows && c.rows == a.rows && c.columns == b.columns) { "gemm: shapes do not conform" }
        if (c.rows == 0 || c.columns == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Gemm, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pb = pins.stage(b, addressing[1])
            val pc = pins.stage(c, addressing[2])
            val fn = symbol(VendorOperation.Gemm).reinterpret<GemmFn>()
            fn(
                layout, transposeFor(addressing[0], layout), transposeFor(addressing[1], layout),
                c.rows, c.columns, a.columns, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
            pc.writeBack()
        } finally {
            pins.release()
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
        require(a.rows == if (rightSide) c.columns else c.rows) { "symm: the symmetric operand has the wrong order" }
        if (c.rows == 0 || c.columns == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Symm, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pb = pins.stage(b, addressing[1])
            val pc = pins.stage(c, addressing[2])
            val fn = symbol(VendorOperation.Symm).reinterpret<SymmFn>()
            fn(
                layout, sideFor(rightSide), uploFor(a, addressing[0], layout), c.rows, c.columns, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
            pc.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun syrk(alpha: Double, a: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "syrk")
        require(c.rows == a.rows) { "syrk: shapes do not conform" }
        if (c.rows == 0) return
        val operands = listOf(a, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Syrk, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pc = pins.stage(c, addressing[1])
            val fn = symbol(VendorOperation.Syrk).reinterpret<SyrkFn>()
            fn(
                layout, uploFor(c, addressing[1], layout), transposeFor(addressing[0], layout),
                c.rows, a.columns, alpha, pa.pointer, pa.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
            pc.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun syr2k(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "syr2k")
        require(c.rows == a.rows && a.rows == b.rows && a.columns == b.columns) { "syr2k: shapes do not conform" }
        if (c.rows == 0) return
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Syr2k, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pb = pins.stage(b, addressing[1])
            val pc = pins.stage(c, addressing[2])
            val fn = symbol(VendorOperation.Syr2k).reinterpret<Syr2kFn>()
            fn(
                layout, uploFor(c, addressing[2], layout), transposeFor(addressing[0], layout),
                c.rows, a.columns, alpha, pa.pointer, pa.leadingDimension,
                pb.pointer, pb.leadingDimension, beta, pc.pointer, pc.leadingDimension,
            )
            pc.writeBack()
        } finally {
            pins.release()
        }
    }

    override fun trmm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean) =
        triangularMatrix(alpha, a, b, rightSide, VendorOperation.Trmm, "trmm")

    override fun trsm(alpha: Double, a: MatrixWindow, b: MatrixWindow, rightSide: Boolean) =
        triangularMatrix(alpha, a, b, rightSide, VendorOperation.Trsm, "trsm")

    private fun triangularMatrix(
        alpha: Double,
        a: MatrixWindow,
        b: MatrixWindow,
        rightSide: Boolean,
        operation: VendorOperation,
        what: String,
    ) {
        requireTriangular(a, what)
        require(if (rightSide) a.rows == b.columns else a.rows == b.rows) { "$what: shapes do not conform" }
        if (b.rows == 0 || b.columns == 0) return
        val operands = listOf(a, b)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(operation, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pb = pins.stage(b, addressing[1])
            val fn = symbol(operation).reinterpret<TriangularMatrixFn>()
            fn(
                layout, sideFor(rightSide), uploFor(a, addressing[0], layout),
                transposeFor(addressing[0], layout), diagFor(a), b.rows, b.columns, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension,
            )
            pb.writeBack()
        } finally {
            pins.release()
        }
    }

    /** Direct where the vendor exports `cblas_dgemmt`, and otherwise the shared composition. */
    override fun gemmt(alpha: Double, a: MatrixWindow, b: MatrixWindow, beta: Double, c: MatrixWindow) {
        requireStructured(c, "gemmt")
        require(a.columns == b.rows && c.rows == a.rows && c.columns == b.columns) { "gemmt: shapes do not conform" }
        if (c.rows == 0 || c.columns == 0) return
        if (VendorOperation.Gemmt !in directlyImplemented) return composeGemmt(alpha, a, b, beta, c)
        val operands = listOf(a, b, c)
        val layout = layoutOf(operands)
        val addressing = effectiveAddressing(VendorOperation.Gemmt, operands)
        val pins = Pins()
        try {
            val pa = pins.stage(a, addressing[0])
            val pb = pins.stage(b, addressing[1])
            val pc = pins.stage(c, addressing[2])
            val fn = symbol(VendorOperation.Gemmt).reinterpret<GemmFn>()
            fn(
                layout, uploFor(c, addressing[2], layout), transposeFor(addressing[0], layout),
                transposeFor(addressing[1], layout), c.rows, a.columns, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
            pc.writeBack()
        } finally {
            pins.release()
        }
    }

    private fun readVersion(): String {
        mklVersion()?.let { return it }
        stringVersion("openblas_get_config")?.let { return it }
        stringVersion("bli_info_get_version_str")?.let { return it }
        return if (vendor == Vendor.Accelerate) "system framework" else "unreported"
    }

    private fun mklVersion(): String? {
        val address = dlsym(handle, "MKL_Get_Version_String") ?: return null
        val fn = address.reinterpret<VersionFn>()
        return memScoped {
            val buffer = allocArray<ByteVar>(VERSION_BUFFER)
            fn(buffer, VERSION_BUFFER)
            buffer.toKString().trim().ifEmpty { null }
        }
    }

    private fun stringVersion(symbol: String): String? {
        val address = dlsym(handle, symbol) ?: return null
        val fn = address.reinterpret<StringFn>()
        return fn()?.toKString()?.trim()?.ifEmpty { null }
    }

    internal companion object {
        private const val VERSION_BUFFER = 256

        /** `MKL_THREADING_SEQUENTIAL`. */
        private const val MKL_SEQUENTIAL = 1

        /** `Dl_info` is four pointers, of which only the first, `dli_fname`, is read. */
        private const val DL_INFO_BYTES = 32

        /**
         * Where a bundled payload for [vendor] would sit beside a Native executable, or nothing when the
         * combination is never bundled.
         *
         * Tried after every installed candidate, because an installed library is the one the operator chose;
         * see [Bundle]. A Native binary has no classpath, so the same layout the JVM module publishes as
         * resources is looked for on disk, relative to the working directory and to the executable itself.
         * A path that does not exist simply fails to open and the search moves on.
         */
        private fun bundled(vendor: Vendor): List<String> {
            val relative = Bundle.path(vendor, hostPlatform()) ?: return emptyList()
            val roots = listOfNotNull(".", executableDirectory())
            return roots.map { "$it/$relative" }
        }

        /** The directory holding this executable, read from the link the kernel maintains. */
        private fun executableDirectory(): String? = memScoped {
            val buffer = allocArray<ByteVar>(PATH_BYTES)
            val length = readlink("/proc/self/exe", buffer, (PATH_BYTES - 1).convert())
            if (length <= 0) return@memScoped null
            buffer[length] = 0
            buffer.toKString().substringBeforeLast('/', "").ifEmpty { null }
        }

        private const val PATH_BYTES = 4096

        /**
         * Opens the first candidate of [vendor] that loads and exports every required CBLAS symbol, or null.
         *
         * A library that opens but is missing part of the surface is rejected rather than half-bound, so a
         * partial install fails here instead of at the first call that needs the missing piece.
         */
        fun open(vendor: Vendor): NativeVendorBlas? {
            val installed = vendor.resolvedCandidates(getenv("HOME")?.toKString())
            for (candidate in installed + bundled(vendor)) {
                val handle = dlopen(candidate, RTLD_NOW) ?: continue
                if (missingRequiredSymbols { dlsym(handle, it) != null }.isNotEmpty()) {
                    // Nothing has been called into it yet, so it can go back the way it came rather than
                    // staying mapped and competing for the global name of a symbol it half-exports.
                    dlclose(handle)
                    continue
                }
                val blas = NativeVendorBlas(vendor, candidate, handle)
                // Ordered: the single-thread configuration is established before the probe, because the probe
                // is arithmetic and oneMKL resolves its threading layer on the first call that reaches it.
                blas.enforceSingleThread()
                if (!blas.verifiedAbi()) continue
                if (!blas.confirmSingleThread()) continue
                return blas
            }
            return null
        }
    }
}

/** Opens the preferred available vendor for this host, or null when none is installed. */
public actual fun openVendorBlas(only: Vendor?): VendorBlas? {
    val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
    return candidates.firstNotNullOfOrNull(NativeVendorBlas::open)
}
