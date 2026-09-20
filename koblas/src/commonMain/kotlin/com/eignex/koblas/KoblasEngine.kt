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
import com.eignex.koblas.dense.DenseTriangularKernels
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.DenseVectorRoute
import com.eignex.koblas.dense.HostDenseBlas
import com.eignex.koblas.dense.PackedMatrix
import com.eignex.koblas.dense.PortableDenseBlas
import com.eignex.koblas.dense.PortablePanelKernels
import com.eignex.koblas.dense.PortableProductKernels
import com.eignex.koblas.dense.PortableTriangularKernels
import com.eignex.koblas.dense.RoutedDenseBlas
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.SPARSE_SCHEDULING
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.sparse.SparsePanelKernels
import com.eignex.koblas.sparse.SparseRoute
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.RouteKind
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
 * An immutable engine providing portable Kotlin BLAS at every level, optionally composing a host library.
 *
 * [vendor] records the installed host binding this engine can reach. It is separately callable for explicit
 * host comparisons and attribution, and on most engines it is nothing else: [BuiltinEngines.scalar] and
 * [BuiltinEngines.simd] never resolve one, so a scalar or JVM SIMD benchmark arm is independent of host
 * libraries by construction. A platform default may also compose it into ordinary dense Level 2 and 3 calls
 * under a fixed policy, which Kotlin/Native's does; holding a binding is not evidence that a given call
 * reached it, and [routeOf] is what answers that for one call.
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
    /** Diagonal-block substitution and the right-hand-side grouping triangular routines schedule with. */
    public val triangularKernels: DenseTriangularKernels = PortableTriangularKernels,
    private val denseBlas: PortableDenseBlas =
        PortableDenseBlas(vectorKernels, panelKernels, productKernels, triangularKernels),
    /**
     * The installed library this engine's ordinary dense Level 2 and 3 calls may be composed from.
     *
     * Separate from [vendor] rather than read off it, because the two answer different questions. A platform
     * default that composes one passes the same binding to both; an engine retaining a binding only so a
     * caller can reach it explicitly passes it as [vendor] alone, and its dense calls stay portable.
     */
    hostDense: Blas? = null,
    private val dense: RoutedDenseBlas = hostDense?.let { HostDenseBlas(denseBlas, it) } ?: denseBlas,
    private val sparseBlas: SparseBlas = SparseAlgorithms(
        vectorKernels,
        indexedSparseKernels,
        SparsePanelKernels(vectorKernels, panelKernels),
    ),
) : DenseBlas by dense,
    SparseBlas by sparseBlas {
    /**
     * The component that owns built-in sparse Level 2 and 3 calls.
     *
     * Always this library's portable CSC scheduling. The Level 1 kernels an individual column reaches are a
     * separate question, and [com.eignex.koblas.sparse.SparseBlas.routeOf] is what answers it.
     */
    public val sparseImplementation: String = SPARSE_SCHEDULING

    /**
     * Short read-only description of what this engine selected, for logs.
     *
     * A selection, not a claim about execution: which of the selected implementations a given call reaches
     * depends on its shape. The [routeOf] overloads describe concrete dense and sparse calls.
     */
    public val name: String
        get() = "${vectorKernels.name}/${sparseKernels.name}/${panelKernels.name}/${productKernels.name}/" +
            triangularKernels.name

    /**
     * What a built-in dense Level 2 or 3 call of this [operation] and shape actually executes.
     *
     * Traversal is this library's own portable code wherever this library schedules the call. Its panels use
     * the selected backend, and a window too short for one falls to the portable body. An engine's name
     * says which backend was selected and nothing about which of its bodies a call ran.
     *
     * Where the platform default composes an installed library, a call with enough arithmetic for it is one
     * whole vendor entry point instead, and the route names which library, which symbol and what this
     * library still did around it. The answer comes from the decision the call itself makes, so an operation
     * with no entry point of its own, such as a product between operands packed for this library's register
     * tile, reports the portable schedule it really runs however large it is.
     */
    public fun routeOf(operation: DenseMatrixOperation, call: DenseCall): DenseMatrixRoute =
        dense.routeOf(operation, call)

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
     * The route for a dense vector call of this [operation] and [length].
     *
     * [contiguous] says whether the operands have unit stride. A composed route means values or aliasing
     * decide which implementation finishes the call; inspecting a route does not execute it.
     */
    public fun routeOf(operation: DenseOperation, length: Int, contiguous: Boolean = true): DenseVectorRoute {
        require(length >= 0) { "negative operation length" }
        val selection = vectorKernels.name
        if (length == 0) {
            return DenseVectorRoute(operation, RouteKind.NoWork, selection, "the vector is empty")
        }
        val reached = vectorKernels.implementationFor(operation, length, contiguous)
            ?: return DenseVectorRoute(
                operation,
                RouteKind.Composed,
                selection,
                "values or aliasing decide which implementation completes the call",
            )
        return if (reached == selection) {
            DenseVectorRoute(operation, RouteKind.Direct, reached, null)
        } else {
            DenseVectorRoute(operation, RouteKind.Delegated, reached, "$selection falls back to $reached")
        }
    }

    /** The route for a sparse vector call over [count] stored entries. */
    public fun routeOf(operation: SparseOperation, count: Int): SparseRoute = sparseKernels.routeOf(operation, count)

    override fun toString(): String = "KoblasEngine($name)"
}

/**
 * Exact built-in implementations for tests and benchmarks.
 *
 * Not a menu of production choices, which is why it is behind [KoblasEngineApi]. [koblas] is what this
 * platform selected, which is a policy: it takes the kernels that have the evidence to be default and the
 * portable ones everywhere else. On the JVM that policy currently arrives at [simd] itself, and holding the
 * two apart is still the point, because which engine a policy lands on is a measurement's conclusion and
 * not something a benchmark arm may assume. [scalar] is not a faster or slower alternative to [simd] at a
 * given size either: it is the floor the vectorised kernels stand on, exposed on its own so a benchmark can
 * time it and a conformance test can compare against it.
 *
 * Kotlin/Native has no Vector API, so [simd] is null there and the selected engine is [scalar] where no
 * host library is installed and a composition over that library where one is.
 */
@KoblasEngineApi
public expect object BuiltinEngines {
    /** Portable Kotlin across all three levels. */
    public val scalar: KoblasEngine

    /**
     * Every JVM Vector API kernel this library owns, or null when the module is unavailable.
     *
     * On the JVM, [koblas] selects this same engine when it is available. A window too short or too
     * strided for a vector body runs a portable fallback; [KoblasEngine.routeOf] identifies the
     * bodies reached by a particular call rather than treating the whole engine as vectorized.
     */
    public val simd: KoblasEngine?
}

/** Optional platform-default Native host binding, resolved once only when that platform policy asks for it. */
internal val selectedVendor: Blas? by lazy { openBlas() }
