@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.vendor

import com.eignex.koblas.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.requireStructured
import com.eignex.koblas.dense.requireTriangular
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.DoubleVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.invoke
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.plus
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import platform.posix.RTLD_NOW
import platform.posix.dlclose
import platform.posix.dlopen
import platform.posix.dlsym
import platform.posix.getenv
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
) : Blas {
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
        if (vendor.threadControl == ThreadControl.Environment) setenv(ACCELERATE_THREAD_LIMIT, "1", 1)
        if (vendor.threadControl == ThreadControl.Mkl) {
            dlsym(handle, "MKL_Set_Threading_Layer")?.reinterpret<SetLayerFn>()?.invoke(MKL_SEQUENTIAL)
            dlsym(handle, "MKL_Set_Dynamic")?.reinterpret<SetThreadsFn>()?.invoke(0)
        }
        val setter = vendor.threadControl.setter ?: return
        dlsym(handle, setter)?.reinterpret<SetThreadsFn>()?.invoke(1)
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
        val control = vendor.threadControl
        val getter = control.getter
        if (getter == null) {
            // Accelerate, held through the environment and unable to answer. Declared, not discovered.
            threadEvidence = ThreadEvidence.Unconfirmed
            return true
        }
        val fn = dlsym(handle, getter)?.reinterpret<GetThreadsFn>()
        if (fn == null) {
            // A build of a known vendor without that vendor's control is not one this code knows; see
            // ThreadControl.absenceMeansSerial for the one reading of an absent control that is positive.
            if (!control.absenceMeansSerial) return false
            threadEvidence = ThreadEvidence.Confirmed
            return true
        }
        if (fn() != 1) return false
        threadEvidence = ThreadEvidence.Confirmed
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
        val fn = dlsym(handle, BlasOperation.Dot.entryPoint)?.reinterpret<DotFn>() ?: return false
        return withPins { pins ->
            val x = AbiProbe.x
            val y = AbiProbe.y
            fn(x.size, pins.pointer(x, 0), 1, pins.pointer(y, 0), 1) == AbiProbe.DOT
        }
    }

    private fun probeGemm(): Boolean {
        val fn = dlsym(handle, BlasOperation.Gemm.entryPoint)?.reinterpret<GemmFn>() ?: return false
        return withPins { pins ->
            val a = AbiProbe.identity
            val b = AbiProbe.operand
            val c = DoubleArray(b.size)
            fn(
                Cblas.COL_MAJOR, Cblas.NO_TRANS, Cblas.NO_TRANS, 2, 2, 2, 1.0,
                pins.pointer(a, 0), 2, pins.pointer(b, 0), 2, 0.0, pins.pointer(c, 0), 2,
            )
            c.indices.all { c[it] == b[it] }
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
        Array(BlasOperation.entries.size) { dlsym(handle, BlasOperation.entries[it].entryPoint) }

    override val directlyImplemented: Set<BlasOperation> =
        BlasOperation.entries.filterTo(LinkedHashSet()) { entryPoints[it.ordinal] != null }

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
        transfer = null,
    )

    private fun symbol(operation: BlasOperation): COpaquePointer = checkNotNull(entryPoints[operation.ordinal]) {
        "${vendor.vendorName} at $libraryPath does not export ${operation.entryPoint}"
    }

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    override fun dot(x: DenseVector, y: DenseVector): Double {
        requireSameSize(x.size, y.size, "dot")
        return rawDot(x.values, x.offset, x.stride, y.values, y.offset, y.stride, x.size)
    }

    override fun nrm2(x: DenseVector): Double = rawNrm2(x.values, x.offset, x.stride, x.size)

    override fun asum(x: DenseVector): Double = rawAsum(x.values, x.offset, x.stride, x.size)

    /*
     * The raw Level 1 surface, which is what the engine's kernels call, and which the [DenseVector] methods
     * above delegate to so that one implementation answers both.
     *
     * It takes a run apart rather than wrapping it because at these widths an object costs more than the
     * arithmetic: `cblas_ddot` over eight entries takes about eight nanoseconds, and each allocation on this
     * runtime is a few. That is also why the no-work question is a comparison against zero here rather than a
     * list handed to a shared helper.
     *
     * Each pin is taken by an inlined `usePinned`, which releases on the way out of its block including
     * through an exception, so an operand is never left pinned by a call that threw.
     */
    internal fun rawDot(
        a: DoubleArray,
        aOffset: Int,
        aStride: Int,
        b: DoubleArray,
        bOffset: Int,
        bStride: Int,
        n: Int,
    ): Double {
        if (n == 0) return 0.0
        val fn = symbol(BlasOperation.Dot).reinterpret<DotFn>()
        a.usePinned { pa ->
            b.usePinned { pb ->
                return fn(
                    n,
                    pa.addressOf(baseIndex(aOffset, aStride, n)),
                    aStride,
                    pb.addressOf(baseIndex(bOffset, bStride, n)),
                    bStride,
                )
            }
        }
    }

    internal fun rawNrm2(a: DoubleArray, offset: Int, stride: Int, n: Int): Double =
        rawReduce(a, offset, stride, n, BlasOperation.Nrm2)

    internal fun rawAsum(a: DoubleArray, offset: Int, stride: Int, n: Int): Double =
        rawReduce(a, offset, stride, n, BlasOperation.Asum)

    private fun rawReduce(a: DoubleArray, offset: Int, stride: Int, n: Int, operation: BlasOperation): Double {
        if (n == 0) return 0.0
        val fn = symbol(operation).reinterpret<ReduceFn>()
        a.usePinned { pinned ->
            return fn(n, pinned.addressOf(baseIndex(offset, stride, n)), abs(stride))
        }
    }

    /** The index the library returns, restored to the caller's order when the stride runs backwards. */
    internal fun rawIamax(a: DoubleArray, offset: Int, stride: Int, n: Int): Int {
        if (n == 0) return 0
        val fn = symbol(BlasOperation.Iamax).reinterpret<IndexFn>()
        a.usePinned { pinned ->
            val found = fn(n, pinned.addressOf(baseIndex(offset, stride, n)), abs(stride))
            return if (stride >= 0) found else n - 1 - found
        }
    }

    @Suppress("LongParameterList") // the BLAS daxpy signature, unwrapped
    internal fun rawAxpy(
        alpha: Double,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        n: Int,
    ) {
        if (n == 0) return
        val fn = symbol(BlasOperation.Axpy).reinterpret<AxpyFn>()
        x.usePinned { px ->
            y.usePinned { py ->
                fn(
                    n,
                    alpha,
                    px.addressOf(baseIndex(xOffset, xStride, n)),
                    xStride,
                    py.addressOf(baseIndex(yOffset, yStride, n)),
                    yStride,
                )
            }
        }
    }

    internal fun rawScal(alpha: Double, a: DoubleArray, offset: Int, stride: Int, n: Int) {
        if (n == 0) return
        val fn = symbol(BlasOperation.Scal).reinterpret<ScalFn>()
        a.usePinned { pinned ->
            fn(n, alpha, pinned.addressOf(baseIndex(offset, stride, n)), abs(stride))
        }
    }

    @Suppress("LongParameterList") // the BLAS dswap signature, unwrapped
    internal fun rawSwap(
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        n: Int,
    ) {
        if (n == 0) return
        val fn = symbol(BlasOperation.Swap).reinterpret<CopyFn>()
        x.usePinned { px ->
            y.usePinned { py ->
                fn(
                    n,
                    px.addressOf(baseIndex(xOffset, xStride, n)),
                    xStride,
                    py.addressOf(baseIndex(yOffset, yStride, n)),
                    yStride,
                )
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS drot signature, unwrapped
    internal fun rawRot(
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        n: Int,
        c: Double,
        s: Double,
    ) {
        if (n == 0) return
        val fn = symbol(BlasOperation.Rot).reinterpret<RotFn>()
        x.usePinned { px ->
            y.usePinned { py ->
                fn(
                    n,
                    px.addressOf(baseIndex(xOffset, xStride, n)),
                    xStride,
                    py.addressOf(baseIndex(yOffset, yStride, n)),
                    yStride,
                    c,
                    s,
                )
            }
        }
    }

    override fun iamax(x: DenseVector): Int = rawIamax(x.values, x.offset, x.stride, x.size)

    override fun axpy(alpha: Double, x: DenseVector, y: DenseVector) {
        requireSameSize(x.size, y.size, "axpy")
        rawAxpy(alpha, x.values, x.offset, x.stride, y.values, y.offset, y.stride, x.size)
    }

    override fun scal(alpha: Double, x: DenseVector) = rawScal(alpha, x.values, x.offset, x.stride, x.size)

    override fun copy(x: DenseVector, y: DenseVector) = twoVector(x, y, BlasOperation.Copy, "copy")

    override fun swap(x: DenseVector, y: DenseVector) {
        requireSameSize(x.size, y.size, "swap")
        rawSwap(x.values, x.offset, x.stride, y.values, y.offset, y.stride, x.size)
    }

    private fun twoVector(x: DenseVector, y: DenseVector, operation: BlasOperation, what: String) {
        requireSameSize(x.size, y.size, what)
        if (noWorkReason(emptyList(), listOf(x)) != null) return
        withPins { pins ->
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(operation).reinterpret<CopyFn>()
            fn(x.size, px.pointer, px.increment, py.pointer, py.increment)
        }
    }

    override fun rot(x: DenseVector, y: DenseVector, c: Double, s: Double) {
        requireSameSize(x.size, y.size, "rot")
        rawRot(x.values, x.offset, x.stride, y.values, y.offset, y.stride, x.size, c, s)
    }

    // Level 2. One matrix operand, passed in place with its own row count as the leading dimension.

    override fun gemv(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        requireGemvOperands(a, transposeA, x.size, y.size)
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(BlasOperation.Gemv).reinterpret<GemvFn>()
            fn(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
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
        requireStructured(structure, "symv")
        requireSymvOperands(a, x.size, y.size)
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(BlasOperation.Symv).reinterpret<SymvFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
        }
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix) {
        requireGerOperands(x.size, y.size, a)
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(BlasOperation.Ger).reinterpret<GerFn>()
            fn(
                Cblas.COL_MAJOR, a.rows, a.cols, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
        }
    }

    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(structure, "syr")
        requireSyrOperands(a, x.size, "syr")
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val fn = symbol(BlasOperation.Syr).reinterpret<SyrFn>()
            fn(
                Cblas.COL_MAJOR,
                uploFor(structure),
                a.rows,
                alpha,
                px.pointer,
                px.increment,
                pa.pointer,
                pa.leadingDimension,
            )
        }
    }

    override fun syr2(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrix, structure: MatrixStructure) {
        requireStructured(structure, "syr2")
        requireSyr2Operands(a, x.size, y.size, "syr2")
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(BlasOperation.Syr2).reinterpret<Syr2Fn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
        }
    }

    override fun trsv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, BlasOperation.Trsv, "trsv")

    override fun trmv(a: DenseMatrix, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, BlasOperation.Trmv, "trmv")

    @Suppress("LongParameterList") // the shared triangular vector signature plus its entry point
    private fun triangularVector(
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
        operation: BlasOperation,
        what: String,
    ) {
        requireTriangular(structure, what)
        requireTriangularVectorOperands(a, x.size, what)
        if (noWorkReason(listOf(a), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val fn = symbol(operation).reinterpret<TriangularVectorFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA), diagFor(structure),
                a.rows, pa.pointer, pa.leadingDimension, px.pointer, px.increment,
            )
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(BlasOperation.Gemm).reinterpret<GemmFn>()
            fn(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                c.rows, c.cols, depth, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        requireStructured(structure, "symm")
        requireSymmOperands(a, b, c, rightSide)
        if (noWorkReason(listOf(c), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(BlasOperation.Symm).reinterpret<SymmFn>()
            fn(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        requireStructured(structure, "syrk")
        requireSyrkOperands(a, transposeA, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWorkReason(listOf(c), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val pc = pins.stage(c)
            val fn = symbol(BlasOperation.Syrk).reinterpret<SyrkFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, pa.pointer, pa.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        requireStructured(structure, "syr2k")
        requireSyr2kOperands(a, b, transposeA, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWorkReason(listOf(c), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(BlasOperation.Syr2k).reinterpret<Syr2kFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, pa.pointer, pa.leadingDimension,
                pb.pointer, pb.leadingDimension, beta, pc.pointer, pc.leadingDimension,
            )
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
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, BlasOperation.Trmm, "trmm")

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun trsm(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, BlasOperation.Trsm, "trsm")

    @Suppress("LongParameterList") // the shared triangular matrix signature plus its entry point
    private fun triangularMatrix(
        alpha: Double,
        a: DenseMatrix,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrix,
        rightSide: Boolean,
        operation: BlasOperation,
        what: String,
    ) {
        requireTriangular(structure, what)
        requireTriangularMatrixOperands(a, b, rightSide, what)
        if (noWorkReason(listOf(b), emptyList()) != null) return
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val fn = symbol(operation).reinterpret<TriangularMatrixFn>()
            fn(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure),
                transposeFor(transposeA), diagFor(structure), b.rows, b.cols, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension,
            )
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
        requireStructured(structure, "gemmt")
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        val depth = if (transposeA) a.rows else a.cols
        if (noWorkReason(listOf(c), emptyList()) != null) return
        if (BlasOperation.Gemmt !in directlyImplemented) {
            return composeGemmt(alpha, a, transposeA, b, transposeB, beta, c, structure)
        }
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(BlasOperation.Gemmt).reinterpret<GemmFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                transposeFor(transposeB), c.rows, depth, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
         * Opens the first candidate of [vendor] that loads and exports every required CBLAS symbol, or null.
         *
         * A library that opens but is missing part of the surface is rejected rather than half-bound, so a
         * partial install fails here instead of at the first call that needs the missing piece.
         */
        fun open(vendor: Vendor): NativeVendorBlas? {
            val installed = vendor.resolvedCandidates(getenv("HOME")?.toKString())
            for (candidate in installed) {
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
public actual fun openBlas(only: Vendor?): Blas? {
    val candidates = only?.let { listOf(it) } ?: Vendor.select(hostPlatform())
    return candidates.firstNotNullOfOrNull(NativeVendorBlas::open)
}
