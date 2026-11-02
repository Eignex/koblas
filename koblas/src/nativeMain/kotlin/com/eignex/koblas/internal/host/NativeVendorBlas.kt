@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.internal.dense.triangularMatrix
import com.eignex.koblas.internal.host.Cblas
import com.eignex.koblas.internal.host.diagFor
import com.eignex.koblas.internal.host.positiveIncrement
import com.eignex.koblas.internal.host.sideFor
import com.eignex.koblas.internal.host.transposeFor
import com.eignex.koblas.internal.host.uploFor
import com.eignex.koblas.vendor.*
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
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
import platform.posix.getenv

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

/**
 * Vendor BLAS reached through the platform loader and typed function pointers.
 *
 * Resolve symbols once from the library's own handle so another loaded BLAS cannot supply them.
 * Pinned operands reach caller storage directly, requiring no transfer adapter in the route, and a borrowed
 * region is pinned at its first entry with its parent's column spacing, so it needs no packing either.
 */
internal class NativeVendorBlas private constructor(
    override val vendor: Vendor,
    private val candidate: String,
    private val library: NativeLibrary,
) : KoblasEngine() {
    override val name: String get() = "${vendor.vendorName.lowercase()}"

    /**
     * File reported by the dynamic loader for the key symbol. A bare soname can resolve to an
     * already-loaded absolute path, so the candidate name alone does not identify the executing file.
     */
    override val libraryPath: String by lazy { library.symbolOwner(vendor.keySymbol) ?: candidate }

    override val version: String by lazy { readVersion() }

    /**
     * Select and verify oneMKL's LP64 interface before any other oneMKL call.
     *
     * The setter overrides an inherited `MKL_INTERFACE_LAYER` and returns the selected interface.
     * A missing setter or a runtime already fixed to another interface must not reach ABI probes.
     */
    private fun establishIntegerInterface(): Boolean {
        if (vendor != Vendor.OneMkl) return true
        val fn = library.symbol("MKL_Set_Interface_Layer")?.reinterpret<SetLayerFn>() ?: return false
        return fn(MKL_LP64) == MKL_LP64
    }

    /**
     * Configure one compute thread after integer interface selection and before any arithmetic.
     *
     * oneMKL must select its sequential layer before first use to avoid resolving a threaded layer
     * that requires an unavailable OpenMP runtime. Thread count and dynamic expansion are fixed too.
     * [confirmSingleThread] checks the result after the ABI probe; configuration precedes the probe
     * because the probe itself can resolve the threading layer.
     */
    private fun enforceSingleThread() {
        // Accelerate has no thread-count entry point; see ACCELERATE_THREAD_LIMIT. Overwrite is on so a value
        // inherited from the launching shell cannot weaken the invariant.
        if (vendor.threadControl == ThreadControl.Environment) {
            setNativeEnvironmentVariable(ACCELERATE_THREAD_LIMIT, "1")
        }
        if (vendor.threadControl == ThreadControl.Mkl) {
            library.symbol("MKL_Set_Threading_Layer")?.reinterpret<SetLayerFn>()?.invoke(MKL_SEQUENTIAL)
            library.symbol("MKL_Set_Dynamic")?.reinterpret<SetThreadsFn>()?.invoke(0)
        }
        val setter = vendor.threadControl.setter ?: return
        library.symbol(setter)?.reinterpret<SetThreadsFn>()?.invoke(1)
    }

    /** Set once at load, after [enforceSingleThread], and never again. */
    override var threadEvidence: ThreadEvidence = ThreadEvidence.Unconfirmed
        private set

    /**
     * Check the configured thread count, rejecting libraries that still report multiple threads.
     * [Vendor.Accelerate] has no read-back entry point and is accepted using its environment control.
     */
    private fun confirmSingleThread(): Boolean {
        val control = vendor.threadControl
        val getter = control.getter
        if (getter == null) {
            // Accelerate, held through the environment and unable to answer. Declared, not discovered.
            threadEvidence = ThreadEvidence.Unconfirmed
            return true
        }
        val fn = library.symbol(getter)?.reinterpret<GetThreadsFn>()
        if (fn == null) {
            // A build of a known vendor without that vendor's control is not one this code knows; see
            // ThreadControl.permitsAbsentControl for the serial-build exception.
            if (!control.permitsAbsentControl(candidate)) return false
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
        val fn = library.symbol(BLIS_INTEGER_WIDTH)?.reinterpret<GetThreadsFn>() ?: return true
        return fn() == LP64_INTEGER_BITS
    }

    private fun probeDot(): Boolean {
        val fn = library.symbol(EngineImplementation.Dot.entryPoint)?.reinterpret<DotFn>() ?: return false
        return withPins { pins ->
            val x = AbiProbe.x
            val y = AbiProbe.y
            fn(x.size, pins.pointer(x, 0), 1, pins.pointer(y, 0), 1) == AbiProbe.DOT
        }
    }

    private fun probeGemm(): Boolean {
        val fn = library.symbol(EngineImplementation.Gemm.entryPoint)?.reinterpret<GemmFn>() ?: return false
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
     * Resolve entry points once: per-call loader locking and lookup would dominate short Level 1
     * calls. The library handle remains alive for the process.
     */
    private val entryPoints: Array<COpaquePointer?> =
        Array(
            EngineImplementation.entries.size,
        ) { EngineImplementation.entries[it].entryPoint.takeIf { it.isNotEmpty() }?.let(library::symbol) }

    override val implementations: Set<EngineImplementation> =
        EngineImplementation.entries.filterTo(LinkedHashSet()) { entryPoints[it.ordinal] != null }

    private fun symbol(operation: EngineImplementation): COpaquePointer = checkNotNull(entryPoints[operation.ordinal]) {
        "${vendor.vendorName} at $libraryPath does not export ${operation.entryPoint}"
    }

    // Level 1. A vector window is always expressible, so these never stage and never compose.

    /*
     * Primitive Level 1 windows reach pinned storage directly, avoiding operand wrappers for short calls.
     *
     * Inlined `usePinned` releases each pin even when a call throws.
     */
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
        val fn = symbol(EngineImplementation.Dot).reinterpret<DotFn>()
        a.usePinned { pa ->
            b.usePinned { pb ->
                return fn(
                    len,
                    pa.addressOf(baseIndex(aOff, aStride, len)),
                    aStride,
                    pb.addressOf(baseIndex(bOff, bStride, len)),
                    bStride,
                )
            }
        }
    }

    override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        rawReduce(v, vOff, vStride, len, EngineImplementation.Nrm2)

    override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double =
        rawReduce(v, vOff, vStride, len, EngineImplementation.Asum)

    private fun rawReduce(a: DoubleArray, offset: Int, stride: Int, n: Int, operation: EngineImplementation): Double {
        if (n == 0) return 0.0
        val fn = symbol(operation).reinterpret<ReduceFn>()
        a.usePinned { pinned ->
            return fn(n, pinned.addressOf(baseIndex(offset, stride, n)), positiveIncrement(stride, n))
        }
    }

    /** The index the library returns, restored to the caller's order when the stride runs backwards. */
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        if (len == 0) return -1
        val fn = symbol(EngineImplementation.Iamax).reinterpret<IndexFn>()
        v.usePinned { pinned ->
            val found = fn(len, pinned.addressOf(baseIndex(vOff, vStride, len)), positiveIncrement(vStride, len))
            return if (vStride >= 0) found else len - 1 - found
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
        val fn = symbol(EngineImplementation.Axpy).reinterpret<AxpyFn>()
        x.usePinned { px ->
            y.usePinned { py ->
                fn(
                    len,
                    alpha,
                    px.addressOf(baseIndex(xOff, xStride, len)),
                    xStride,
                    py.addressOf(baseIndex(yOff, yStride, len)),
                    yStride,
                )
            }
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (len == 0) return
        val fn = symbol(EngineImplementation.Scale).reinterpret<ScalFn>()
        v.usePinned { pinned ->
            fn(len, alpha, pinned.addressOf(baseIndex(vOff, vStride, len)), positiveIncrement(vStride, len))
        }
    }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (len == 0) return
        val fn = symbol(EngineImplementation.Copy).reinterpret<CopyFn>()
        a.usePinned { px ->
            b.usePinned { py ->
                fn(
                    len,
                    px.addressOf(baseIndex(aOff, aStride, len)),
                    aStride,
                    py.addressOf(baseIndex(bOff, bStride, len)),
                    bStride,
                )
            }
        }
    }

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) {
        if (len == 0) return
        val fn = symbol(EngineImplementation.Swap).reinterpret<CopyFn>()
        a.usePinned { px ->
            b.usePinned { py ->
                fn(
                    len,
                    px.addressOf(baseIndex(aOff, aStride, len)),
                    aStride,
                    py.addressOf(baseIndex(bOff, bStride, len)),
                    bStride,
                )
            }
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
        if (x === y && xOff == yOff && xStride == yStride) {
            PortableEngine.rot(x, xOff, y, yOff, len, c, s, xStride, yStride)
        } else {
            if (len == 0) return
            val fn = symbol(EngineImplementation.Rot).reinterpret<RotFn>()
            x.usePinned { px ->
                y.usePinned { py ->
                    fn(
                        len,
                        px.addressOf(baseIndex(xOff, xStride, len)),
                        xStride,
                        py.addressOf(baseIndex(yOff, yStride, len)),
                        yStride,
                        c,
                        s,
                    )
                }
            }
        }
    }

    // Level 2. One matrix operand, passed in place with its own row count as the leading dimension.

    override fun computeGemv(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
    ) {
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(EngineImplementation.Gemv).reinterpret<GemvFn>()
            fn(
                Cblas.COL_MAJOR, transposeFor(transposeA), a.rows, a.cols, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
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
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(EngineImplementation.Symv).reinterpret<SymvFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                pa.pointer, pa.leadingDimension, px.pointer, px.increment, beta, py.pointer, py.increment,
            )
        }
    }

    override fun computeGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) {
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(EngineImplementation.Ger).reinterpret<GerFn>()
            fn(
                Cblas.COL_MAJOR, a.rows, a.cols, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
        }
    }

    override fun computeSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) {
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val fn = symbol(EngineImplementation.Syr).reinterpret<SyrFn>()
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

    override fun computeSyr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
    ) {
        withPins { pins ->
            val pa = pins.stage(a)
            val px = pins.stage(x)
            val py = pins.stage(y)
            val fn = symbol(EngineImplementation.Syr2).reinterpret<Syr2Fn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
        }
    }

    override fun computeTrsv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, EngineImplementation.Trsv)

    override fun computeTrmv(a: DenseMatrixRegion, structure: MatrixStructure, transposeA: Boolean, x: DenseVector) =
        triangularVector(a, structure, transposeA, x, EngineImplementation.Trmv)

    @Suppress("LongParameterList") // the shared triangular vector signature plus its entry point
    private fun triangularVector(
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
        operation: EngineImplementation,
    ) {
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

    // Level 3. Every operand is column-major with unit row stride, so the layout is fixed and a transpose is a flag.

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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Gemm).reinterpret<GemmFn>()
            fn(
                Cblas.COL_MAJOR, transposeFor(transposeA), transposeFor(transposeB),
                c.rows, c.cols, depth, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Symm).reinterpret<SymmFn>()
            fn(
                Cblas.COL_MAJOR, sideFor(rightSide), uploFor(structure), c.rows, c.cols, alpha,
                pa.pointer, pa.leadingDimension, pb.pointer, pb.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Syrk).reinterpret<SyrkFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, pa.pointer, pa.leadingDimension, beta,
                pc.pointer, pc.leadingDimension,
            )
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Syr2k).reinterpret<Syr2kFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, pa.pointer, pa.leadingDimension,
                pb.pointer, pb.leadingDimension, beta, pc.pointer, pc.leadingDimension,
            )
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
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, EngineImplementation.Trmm)

    @Suppress("LongParameterList") // the BLAS dtrsm signature
    override fun computeTrsm(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
    ) = triangularMatrix(alpha, a, structure, transposeA, b, rightSide, EngineImplementation.Trsm)

    @Suppress("LongParameterList") // the shared triangular matrix signature plus its entry point
    private fun triangularMatrix(
        alpha: Double,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        rightSide: Boolean,
        operation: EngineImplementation,
    ) {
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Gemmt).reinterpret<GemmFn>()
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
        val address = library.symbol("MKL_Get_Version_String") ?: return null
        val fn = address.reinterpret<VersionFn>()
        return memScoped {
            val buffer = allocArray<ByteVar>(VERSION_BUFFER)
            fn(buffer, VERSION_BUFFER)
            buffer.toKString().trim().ifEmpty { null }
        }
    }

    private fun stringVersion(symbol: String): String? {
        val address = library.symbol(symbol) ?: return null
        val fn = address.reinterpret<StringFn>()
        return fn()?.toKString()?.trim()?.ifEmpty { null }
    }

    internal companion object {
        private const val VERSION_BUFFER = 256

        /** `MKL_THREADING_SEQUENTIAL`. */
        private const val MKL_SEQUENTIAL = 1

        /**
         * Open the first [vendor] candidate with every required CBLAS symbol, or null. Reject partial
         * installs before arithmetic; the ABI and single-thread checks must also succeed.
         */
        fun open(vendor: Vendor): NativeVendorBlas? {
            val installed = vendor.loadCandidates(
                (getenv("HOME") ?: getenv("USERPROFILE"))?.toKString(),
                hostPlatform().operatingSystem,
                getenv("PATH")?.toKString(),
            )
            for (candidate in installed) {
                val library = NativeLibrary.open(candidate) ?: continue
                if (missingRequiredSymbols { library.symbol(it) != null }.isNotEmpty()) {
                    // No calls have used this handle; unload the partial library to avoid symbol conflicts.
                    library.close()
                    continue
                }
                val blas = NativeVendorBlas(vendor, candidate, library)
                // oneMKL interface selection must precede dynamic/thread controls, version reads and probes.
                if (!blas.establishIntegerInterface()) continue
                blas.enforceSingleThread()
                if (!blas.verifiedAbi()) continue
                if (!blas.confirmSingleThread()) continue
                return blas
            }
            return null
        }
    }
}
