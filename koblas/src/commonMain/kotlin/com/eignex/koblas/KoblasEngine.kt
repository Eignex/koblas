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

/** JVM SIMD when available; Native host composition when available; portable Kotlin otherwise. */
internal expect fun platformEngine(): KoblasEngine

/**
 * Immutable dense and sparse BLAS with portable fallbacks.
 *
 * [koblas] is the platform default. [BuiltinEngines] provides exact portable and JVM SIMD compositions
 * for tests and benchmarks, neither of which loads a host library. On Native, the default may compose
 * [vendor] into eligible calls. The [routeOf] overloads describe which implementations a call reaches.
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
     * The route for a dense matrix [operation] with the supplied [call] facts.
     *
     * Names the panels and kernels reached by the shared schedule, including short-window fallbacks.
     * A whole-call host route identifies the library and entry point. Products over retained packed
     * operands use this engine's product kernels, since host libraries cannot read that layout.
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
     * [contiguous] says whether the operands have unit stride. A composed route means values
     * decide which implementation finishes the call; inspecting a route does not execute it.
     */
    public fun routeOf(
        operation: DenseOperation,
        length: Int,
        contiguous: Boolean = true,
    ): VectorRoute<DenseOperation> {
        require(length >= 0) { "negative operation length" }
        return vectorRoute(
            operation,
            operation.name.lowercase(),
            vectorKernels.name,
            vectorKernels.implementationFor(operation, length, contiguous),
        )
    }

    /** The route for a sparse vector call over [count] stored entries. */
    public fun routeOf(operation: SparseOperation, count: Int): VectorRoute<SparseOperation> =
        sparseKernels.routeOf(operation, count)

    override fun toString(): String = "KoblasEngine($name)"
}

/**
 * Exact portable and JVM SIMD engines for tests and benchmarks, without changing the default [koblas].
 *
 * The JVM default currently selects [simd] when available and [scalar] otherwise. Native has no SIMD
 * engine and composes an installed host library with portable fallbacks, or uses [scalar] without one.
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
