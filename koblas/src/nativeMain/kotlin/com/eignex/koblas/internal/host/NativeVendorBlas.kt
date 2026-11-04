@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("TooManyFunctions", "LongParameterList") // the CBLAS double-precision surface

package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CFunction
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.DoubleVar
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
private typealias TriangularVectorFn = CFunction<(Int, Int, Int, Int, Int, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias GemmFn =
    CFunction<(Int, Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias SymmFn = CFunction<(Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias SyrkFn = CFunction<(Int, Int, Int, Int, Int, Double, Ptr?, Int, Double, Ptr?, Int) -> Unit>
private typealias TriangularMatrixFn =
    CFunction<(Int, Int, Int, Int, Int, Int, Int, Double, Ptr?, Int, Ptr?, Int) -> Unit>
private typealias BufferFn = CFunction<(Bytes?, Int) -> Unit>
private typealias StringFn = CFunction<() -> Bytes?>
private typealias IntToIntFn = CFunction<(Int) -> Int>
private typealias IntToUnitFn = CFunction<(Int) -> Unit>
private typealias ToIntFn = CFunction<() -> Int>

/** Bit [Vendor.ordinal] is set once this thread has applied that vendor's one-thread setter. */
@kotlin.native.concurrent.ThreadLocal
private var heldVendors: Int = 0

/**
 * Vendor BLAS reached through the platform loader and typed function pointers.
 *
 * Resolve symbols once from the library's own handle so another loaded BLAS cannot supply them.
 * Pinned operands reach caller storage directly, requiring no transfer adapter in the route, and a borrowed
 * region is pinned at its first entry with its parent's column spacing, so it needs no packing either.
 */
internal class NativeVendorBlas private constructor(
    override val vendor: Vendor,
    override val candidate: String,
    private val library: NativeLibrary,
) : HostBlas(),
    AdmissionCalls {
    override val name: String get() = "${vendor.vendorName.lowercase()}"

    /**
     * File reported by the dynamic loader for the key symbol. A bare soname can resolve to an
     * already-loaded absolute path, so the candidate name alone does not identify the executing file.
     */
    override val libraryPath: String by lazy { library.symbolOwner(vendor.keySymbol) ?: candidate }

    override val version: String by lazy { readVersion() }

    override fun callIntToInt(name: String, argument: Int): Int? =
        library.symbol(name)?.reinterpret<IntToIntFn>()?.invoke(argument)

    override fun callIntToUnit(name: String, argument: Int) {
        library.symbol(name)?.reinterpret<IntToUnitFn>()?.invoke(argument)
    }

    override fun callToInt(name: String): Int? = library.symbol(name)?.reinterpret<ToIntFn>()?.invoke()

    override fun callIntoBuffer(name: String, size: Int): String? {
        val fn = library.symbol(name)?.reinterpret<BufferFn>() ?: return null
        return memScoped {
            val buffer = allocArray<ByteVar>(size)
            fn(buffer, size)
            buffer.toKString()
        }
    }

    override fun callToString(name: String): String? =
        library.symbol(name)?.reinterpret<StringFn>()?.invoke()?.toKString()

    override fun setEnvironmentVariable(name: String, value: String) = setNativeEnvironmentVariable(name, value)

    override fun holdThisThread() {
        threadSetter?.invoke(1)
        heldVendors = heldVendors or (1 shl vendor.ordinal)
    }

    private val threadSetter = vendor.threadControl.setter?.let(library::symbol)?.reinterpret<IntToUnitFn>()

    /**
     * Apply the one-thread setter on the calling thread, once per thread.
     *
     * OpenMP keeps its thread count per thread, as do current BLIS builds, and an OpenMP build of OpenBLAS
     * resynchronises its count from the calling thread's. A setting made only on the thread that opened the
     * library would leave every other caller with a full team, so each thread takes the setting before its
     * first call.
     */
    private fun holdCallingThread() {
        if (heldVendors and (1 shl vendor.ordinal) == 0) holdThisThread()
    }

    /** Set once at load by [admit], and never again. */
    override var threadEvidence: ThreadEvidence = ThreadEvidence.Unconfirmed
        private set

    override fun knownAnswersMatch(): Boolean = probeDot() && probeGemm()

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
        EngineImplementation.entries.filterTo(mutableSetOf()) { entryPoints[it.ordinal] != null }

    private fun symbol(operation: EngineImplementation): COpaquePointer {
        holdCallingThread()
        return checkNotNull(entryPoints[operation.ordinal]) {
            "${vendor.vendorName} at $libraryPath does not export ${operation.entryPoint}"
        }
    }

    // Level 1 windows call the selected library directly; alias adaptation preserves that provider.

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
                    pa.runAddress(aOff, aStride, len),
                    aStride,
                    pb.runAddress(bOff, bStride, len),
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
            return fn(n, pinned.runAddress(offset, stride, n), positiveIncrement(stride, n))
        }
    }

    /** The index the library returns, restored to the caller's order when the stride runs backwards. */
    override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
        if (len == 0) return -1
        val fn = symbol(EngineImplementation.Iamax).reinterpret<IndexFn>()
        v.usePinned { pinned ->
            val found = fn(len, pinned.runAddress(vOff, vStride, len), positiveIncrement(vStride, len))
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
                    px.runAddress(xOff, xStride, len),
                    xStride,
                    py.runAddress(yOff, yStride, len),
                    yStride,
                )
            }
        }
    }

    override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
        if (len == 0) return
        val fn = symbol(EngineImplementation.Scale).reinterpret<ScalFn>()
        v.usePinned { pinned ->
            fn(len, alpha, pinned.runAddress(vOff, vStride, len), positiveIncrement(vStride, len))
        }
    }

    override fun copy(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        pair(EngineImplementation.Copy, a, aOff, b, bOff, len, aStride, bStride)

    override fun swap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int) =
        pair(EngineImplementation.Swap, a, aOff, b, bOff, len, aStride, bStride)

    /** `cblas_dcopy` or `cblas_dswap`, which share one signature. */
    private fun pair(
        operation: EngineImplementation,
        a: DoubleArray,
        aOff: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        aStride: Int,
        bStride: Int,
    ) {
        if (len == 0) return
        val fn = symbol(operation).reinterpret<CopyFn>()
        a.usePinned { px ->
            b.usePinned { py ->
                fn(
                    len,
                    px.runAddress(aOff, aStride, len),
                    aStride,
                    py.runAddress(bOff, bStride, len),
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
        if (len == 0) return
        val aliased = x === y && xOff == yOff && xStride == yStride
        val stableX = if (aliased) DoubleArray(len) { x[xOff + it * xStride] } else x
        val stableOffset = if (aliased) 0 else xOff
        val stableStride = if (aliased) 1 else xStride
        val fn = symbol(EngineImplementation.Rot).reinterpret<RotFn>()
        stableX.usePinned { px ->
            y.usePinned { py ->
                fn(
                    len,
                    px.runAddress(stableOffset, stableStride, len),
                    stableStride,
                    py.runAddress(yOff, yStride, len),
                    yStride,
                    c,
                    s,
                )
            }
        }
    }

    // Level 2. One matrix operand, passed in place with its own row count as the leading dimension.

    override fun cblasGemv(
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

    override fun cblasSymv(
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

    override fun cblasGer(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion) {
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

    override fun cblasSyr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, structure: MatrixStructure) {
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

    override fun cblasSyr2(
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
            val fn = symbol(EngineImplementation.Syr2).reinterpret<GerFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), a.rows, alpha,
                px.pointer, px.increment, py.pointer, py.increment, pa.pointer, pa.leadingDimension,
            )
        }
    }

    override fun cblasTriangularVector(
        operation: EngineImplementation,
        a: DenseMatrixRegion,
        structure: MatrixStructure,
        transposeA: Boolean,
        x: DenseVector,
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

    override fun cblasSymm(
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

    override fun cblasSyrk(
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
        withPins { pins ->
            val pa = pins.stage(a)
            val pb = pins.stage(b)
            val pc = pins.stage(c)
            val fn = symbol(EngineImplementation.Syr2k).reinterpret<SymmFn>()
            fn(
                Cblas.COL_MAJOR, uploFor(structure), transposeFor(transposeA),
                c.rows, depth, alpha, pa.pointer, pa.leadingDimension,
                pb.pointer, pb.leadingDimension, beta, pc.pointer, pc.leadingDimension,
            )
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

    internal companion object {
        /**
         * Open the first [vendor] candidate that passes the LP64 probes and single-thread checks.
         * Missing non-probe exports remain absent capabilities.
         */
        fun open(vendor: Vendor): NativeVendorBlas? {
            val installed = vendor.loadCandidates(
                (getenv("HOME") ?: getenv("USERPROFILE"))?.toKString(),
                hostPlatform().operatingSystem,
                getenv("PATH")?.toKString(),
            )
            for (candidate in installed) {
                val library = NativeLibrary.open(candidate) ?: continue
                val blas = NativeVendorBlas(vendor, candidate, library)
                blas.threadEvidence = blas.admit() ?: continue
                return blas
            }
            return null
        }
    }
}
