@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseMatrixRoute
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseProductKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.PackedMatrix
import com.eignex.koblas.dense.PortableDenseBlas
import com.eignex.koblas.dense.PortablePanelKernels
import com.eignex.koblas.dense.PortableProductKernels
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.SPARSE_SCHEDULING
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparsePanelKernels
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
 * operands into native memory and so costs a pass over the data before any arithmetic happens. They are the
 * reductions only: an elementwise loop is vectorised by the JIT without being written in lanes, and a
 * hand-written one measured no faster. On Kotlin/Native there is no Vector API, the portable loops do not
 * vectorise and pay a safepoint poll and a bounds check per element, and the binding pins the caller's array
 * and passes it in place: so the library is the arm, above the width where its per-call cost is paid for.
 *
 * Either way the portable kernels are what the chosen arm calls below its own threshold, not a third engine
 * beside it.
 */
internal expect fun platformEngine(): KoblasEngine

/**
 * An immutable engine providing portable Kotlin BLAS at every level.
 *
 * [vendor] records the separately callable installed host binding; it is not the implementation of built-in
 * Level 2 and 3 calls. This distinction keeps scalar and JVM SIMD benchmark arms independent of host libraries.
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
    /** Optional installed host binding, retained for explicit host comparisons and attribution. */
    public val vendor: Blas?,
    /** Panel arithmetic and the local execution grouping that dense and sparse algorithms schedule with. */
    public val panelKernels: DensePanelKernels = PortablePanelKernels,
    /** Register-tile product arithmetic and the tile geometry dense products pack for. */
    public val productKernels: DenseProductKernels = PortableProductKernels,
    private val denseBlas: PortableDenseBlas = PortableDenseBlas(vectorKernels, panelKernels, productKernels),
    private val sparseBlas: SparseBlas = SparseAlgorithms(
        vectorKernels,
        indexedSparseKernels,
        SparsePanelKernels(vectorKernels, panelKernels),
    ),
) : DenseBlas by denseBlas,
    SparseBlas by sparseBlas {
    /**
     * The component that owns built-in sparse Level 2 and 3 calls.
     *
     * Always this library's portable CSC scheduling. The Level 1 kernels an individual column reaches are a
     * separate question, and [com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] is what answers it.
     */
    public val sparseImplementation: String = SPARSE_SCHEDULING

    /**
     * Short read-only description of what this engine selected, for logs.
     *
     * A selection, not a claim about execution: which of the selected implementations a given call reaches
     * depends on its shape, and [denseRouteOf] and [com.eignex.koblas.sparse.SparseBlas.matrixRouteOf] are
     * what answer that.
     */
    public val name: String
        get() = "${vectorKernels.name}/${sparseKernels.name}/${panelKernels.name}/${productKernels.name}"

    /**
     * What a built-in dense Level 2 or 3 call of this [operation] and shape actually executes.
     *
     * The dense counterpart of [com.eignex.koblas.sparse.SparseBlas.matrixRouteOf]. Traversal is this
     * library's own portable code on every engine; the panels a window reaches are the selected backend's,
     * and a window too short for one falls to the portable body. An engine's name says which backend was
     * selected and nothing about which of its bodies a call ran.
     */
    public fun denseRouteOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute =
        denseBlas.routeOf(operation, call)

    /**
     * `op(A)` copied into the grouped layout this engine's product tile reads, for the left of a product.
     *
     * The copy a product would make per call, made once and kept. Worth it for an operand that is multiplied
     * several times; a single product packs what it needs itself and this only moves that cost earlier.
     *
     * The result owns its storage and is independent of [a] afterwards. It is usable by an engine whose
     * product tile has the same geometry as this one's and refused by any other, which
     * [com.eignex.koblas.dense.PackedLayout] settles before a product writes anything.
     */
    public fun packLeft(a: DenseMatrix, transpose: Boolean = false): PackedMatrix = denseBlas.packLeft(a, transpose)

    /** `op(B)` packed for the right of a product, the counterpart of [packLeft]. */
    public fun packRight(b: DenseMatrix, transpose: Boolean = false): PackedMatrix = denseBlas.packRight(b, transpose)

    /** `C = alpha · A · B + beta · C` between two retained panels, which copies nothing. */
    public fun gemm(alpha: Double, a: PackedMatrix, b: PackedMatrix, beta: Double, c: DenseMatrix): Unit =
        denseBlas.gemm(alpha, a, b, beta, c)

    /**
     * [gemm] with the left operand retained and the right packed for this call.
     *
     * [workspace] lends the right operand's packed panels and the staging a call takes when [b] shares [c].
     */
    @Suppress("LongParameterList") // the product, the transpose of the operand being packed, and the scratch
    public fun gemm(
        alpha: Double,
        a: PackedMatrix,
        b: DenseMatrix,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    ): Unit = denseBlas.gemm(alpha, a, b, transposeB, beta, c, workspace)

    /** [gemm] with the right operand retained and the left packed for this call. */
    @Suppress("LongParameterList") // the product, the transpose of the operand being packed, and the scratch
    public fun gemm(
        alpha: Double,
        a: DenseMatrix,
        transposeA: Boolean,
        b: PackedMatrix,
        beta: Double,
        c: DenseMatrix,
        workspace: Workspace? = null,
    ): Unit = denseBlas.gemm(alpha, a, transposeA, b, beta, c, workspace)

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
 * Exact built-in implementations for tests and benchmarks.
 *
 * Not a menu of production choices, which is why it is behind [KoblasEngineApi]. [koblas] is what this
 * platform selected, which is a policy: it takes the kernels that have the evidence to be default and the
 * portable ones everywhere else, so it need not be either engine here. [scalar] is not a faster or slower
 * alternative to [simd] at a given size either: it is the floor the vectorised kernels stand on, exposed on
 * its own so a benchmark can time it and a conformance test can compare against it.
 *
 * Kotlin/Native has no Vector API, so [simd] is null there and [koblas] is [scalar].
 */
@KoblasEngineApi
public expect object BuiltinEngines {
    /** Portable Kotlin across all three levels. */
    public val scalar: KoblasEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * Not necessarily what [koblas] selected. A kernel is activated by default once it has the evidence for
     * it, and the Level 2 panels do not yet, so on this platform the default is this arm's Level 1 with the
     * portable panels. Which of them a given call reaches is what [KoblasEngine.denseRouteOf] answers.
     */
    public val simd: KoblasEngine?
}

/** Optional platform-default Native host binding, resolved once only when that platform policy asks for it. */
internal val selectedVendor: Blas? by lazy { openBlas() }
