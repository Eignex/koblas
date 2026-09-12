@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.NativeCatalog
import com.eignex.koblas.sparse.IndexedSparseKernels
import com.eignex.koblas.sparse.SparseAlgorithms
import com.eignex.koblas.sparse.SparseBlas
import com.eignex.koblas.sparse.SparseKernels
import com.eignex.koblas.sparse.SparsePanelKernels

/** The immutable platform-selected BLAS engine used by top-level convenience operations. */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine = run {
    val simd = BuiltinEngines.simd
    val c = BuiltinEngines.c
    if (simd != null && c?.nativeVariant != null) {
        densePolicyEngine(simd, BuiltinEngines.exactC(c.nativeVariant), RuntimeCompetitor.JvmVector)
    } else {
        simd ?: c ?: BuiltinEngines.scalar
    }
}

/**
 * An immutable dense and sparse BLAS engine.
 *
 * The default [koblas] instance is selected once for the platform. Tests and benchmarks can construct an
 * exact scalar, C, or SIMD composition from [BuiltinEngines] without changing process-global state. Each
 * composition binds its dense vector, dense panel, packed tile, indexed sparse, and sparse panel kernels once.
 */
public class KoblasEngine internal constructor(
    /** Standalone contiguous dense-vector kernels. */
    public val vectorKernels: DenseVectorKernels,
    /** Dense matrix-panel arithmetic kernels. */
    public val panelKernels: DensePanelKernels,
    /** Packed layout shape and tile arithmetic kernels. */
    public val packedKernels: PackedKernels,
    /** Sparse-vector kernels used by sparse convenience operations. */
    public val sparseKernels: SparseKernels,
    internal val indexedSparseKernels: IndexedSparseKernels,
    /** Ordinary C variant bound to native components; policy calls may retain in-runtime arithmetic. */
    public val nativeVariant: NativeVariant? = null,
    internal val dispatch: DenseDispatch? = null,
    internal val runtimeDescription: ((DenseOperation, Int) -> String)? = null,
) : DenseBlas by BuiltinBlas(vectorKernels, panelKernels, packedKernels),
    SparseBlas by SparseAlgorithms(
        vectorKernels,
        indexedSparseKernels,
        SparsePanelKernels(vectorKernels, panelKernels),
    ) {
    /** Packed panel operations bound to this engine's exact packed kernels. */
    public val packedPanels: PackedPanels = PackedPanels(packedKernels)

    /** Short read-only implementation description for logs and benchmark attribution. */
    public val name: String get() = "built-in/${vectorKernels.name}/${sparseKernels.name}"

    /**
     * Describes the selected implementation without performing arithmetic. [length] is the vector/panel
     * length, product-tile depth, or standalone solve order. Semantic no-work exits still precede execution.
     * Native IDs and layouts identify the selected component. Runtime descriptions identify scalar stages
     * and potential data-dependent or stride fallbacks; inspecting length does not inspect operand values.
     */
    public fun explain(operation: DenseOperation, length: Int): String {
        require(length >= 0) { "negative operation length" }
        dispatch?.let { return it.explain(operation, length) }
        if (nativeVariant != null) {
            val kernel = NativeCatalog.kernels.single {
                it.variant == nativeVariant.id && it.operation == operation.nativeOperation
            }
            return describeNative(kernel)
        }
        runtimeDescription?.let { return it(operation, length) }
        return if (operation.packed) {
            "${vectorKernels.name} packed ${packedKernels.gemmTileRows}x${packedKernels.gemmTileCols}"
        } else {
            vectorKernels.name
        }
    }

    /** Malformed performance overrides retained as read-only diagnostic messages. */
    public val tuningDiagnostics: List<String>
        get() = (dispatch?.profile ?: DenseProfiles.conservative).diagnostics

    override fun toString(): String = "KoblasEngine($name)"
}

/** Built-in engines for implementation comparisons. */
public expect object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public val scalar: KoblasEngine

    /** Compiled C kernels, or null when they are unavailable. */
    public val c: KoblasEngine?

    /** Native variants compiled into this artifact and usable on this host. */
    public val nativeVariants: List<NativeVariant>

    /**
     * Dense C execution at the exact [variant], bypassing performance thresholds.
     * Semantic early exits remain in force. Sparse policy and portable transformation generation retain
     * their separately identified components. Throws if the selected variant is unavailable.
     */
    public fun exactC(variant: NativeVariant): KoblasEngine

    /** JVM Vector API kernels, or null when the Vector API module is unavailable or on a non-JVM target. */
    public val simd: KoblasEngine?
}
