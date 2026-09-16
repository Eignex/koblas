@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.VendorDenseBlas
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.openBlas

/**
 * The immutable platform-selected BLAS engine used by top-level convenience operations.
 *
 * Deferred rather than computed while this file's class initializes. Picking an engine reads
 * [selectedVendor], which is declared below and is itself deferred; resolving eagerly would read that
 * property's backing delegate before the initializer reached it and select against a null vendor.
 */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine by lazy { platformEngine() }

/**
 * The Level 1 arm this platform prefers, which is not the same arm on both.
 *
 * On the JVM the Vector API kernels win at every width, because reaching the library there copies both
 * operands into native memory and so costs a pass over the data before any arithmetic happens. On
 * Kotlin/Native there is no Vector API, the portable loops do not vectorise and pay a safepoint poll and a
 * bounds check per element, and the binding pins the caller's array and passes it in place: so the library is
 * the arm, above the width where its per-call cost is paid for.
 *
 * Either way the portable kernels are what the chosen arm calls below its own threshold, not a third engine
 * beside it.
 */
internal expect fun platformEngine(): KoblasEngine

/**
 * An immutable engine: Kotlin Level 1 beside the vendor BLAS that serves Level 2 and 3.
 *
 * The split is deliberate. Level 1 is arithmetic over one run, where a foreign call costs more than the work,
 * so it stays here and keeps working on a host with no library installed. Level 2 and 3 are whole operations a
 * tuned library does far better than portable code, so they go to the vendor and have no fallback: an
 * accelerator-dependent call on a host without a library raises rather than quietly computing something slower
 * under the same name.
 *
 * Selected once for the platform and immutable afterwards. [BuiltinEngines] constructs exact scalar or SIMD
 * compositions for tests and benchmarks without touching process-global state.
 */
public class KoblasEngine internal constructor(
    /** Contiguous and strided dense Level 1 kernels. */
    public val vectorKernels: DenseVectorKernels,
    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels,
    internal val indexedSparseKernels: IndexedSparseKernels,
    /**
     * The library serving Level 2 and 3, or null on a host where none was found.
     *
     * Public because attribution needs it: a benchmark asks the binding what a concrete call does, and the
     * answer has to come from the same object that will run it rather than from the engine's name.
     */
    public val vendor: Blas?,
) : DenseBlas by VendorDenseBlas(vendor) {
    /** Short read-only implementation description for logs and benchmark attribution. */
    public val name: String
        get() = "${vectorKernels.name}/${sparseKernels.name}/${vendor?.vendor?.vendorName ?: "no vendor"}"

    /**
     * The Level 1 implementation a call of this [operation] and [length] reaches, or null when its own values
     * decide and no width settles it.
     *
     * [contiguous] is part of the question rather than a detail of it: a vector kernel loads a lane block from
     * consecutive elements, so a strided run is scalar work whatever the width.
     */
    public fun explain(operation: DenseOperation, length: Int, contiguous: Boolean = true): String? {
        require(length >= 0) { "negative operation length" }
        return vectorKernels.implementationFor(operation, length, contiguous)
    }

    override fun toString(): String = "KoblasEngine($name)"
}

/**
 * The Level 1 implementations, each beside the selected vendor, for measuring one against another.
 *
 * Not a menu of production choices, which is why it is behind [KoblasEngineApi]. [koblas] is the engine this
 * platform selected, and on the JVM that is [simd], whose kernels already delegate to the portable ones below
 * their lane width and for any strided run. So [scalar] is not a faster or slower alternative at a given
 * size: it is the floor the vectorised kernels stand on, exposed on its own so a benchmark can time it and a
 * conformance test can compare against it.
 *
 * Kotlin/Native has no Vector API, so [simd] is null there and [koblas] is [scalar].
 */
@KoblasEngineApi
public expect object BuiltinEngines {
    /** The portable Kotlin Level 1 kernels beside the selected vendor. */
    public val scalar: KoblasEngine

    /** JVM Vector API Level 1, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: KoblasEngine?
}

/** The vendor every built-in engine shares, resolved once. */
internal val selectedVendor: Blas? by lazy { openBlas() }
