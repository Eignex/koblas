package com.eignex.koblas.dense

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.NativeCatalog
import com.eignex.koblas.internal.kernels.NativeContext
import com.eignex.koblas.internal.kernels.NativeKernel
import com.eignex.koblas.internal.kernels.NativeProbe
import com.eignex.koblas.internal.kernels.NativeRecords
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Immutable operation choices over compatible runtime and exact native components. */
internal class DenseDispatch(
    private val runtime: KoblasEngine,
    private val native: KoblasEngine,
    private val competitor: RuntimeCompetitor,
    val profile: DenseProfile,
) {
    private val compatiblePacked = runtime.packedKernels.gemmTileRows == native.packedKernels.gemmTileRows &&
        runtime.packedKernels.gemmTileCols == native.packedKernels.gemmTileCols
    private val selections = DenseOperation.entries.map { operation ->
        val kernel = NativeCatalog.kernels.singleOrNull {
            it.variant == native.nativeVariant?.id && it.operation == operation.nativeOperation
        }
        NativeSelection(
            if (operation.packed && !compatiblePacked) null else kernel,
            NativeCatalog.host,
            profile[operation],
        ) {
            if (kernel != null &&
                (kernel.state.process != 0 || kernel.state.thread != 0 || kernel.executionMode == 2)
            ) {
                NativeProbe.query(3)?.let(NativeRecords::context) ?: EMPTY_CONTEXT
            } else {
                EMPTY_CONTEXT
            }
        }
    }.toTypedArray()

    fun usesNative(operation: DenseOperation, length: Int): Boolean =
        selections[operation.ordinal].auto(competitor, length) != null

    fun hasNativeChoice(operation: DenseOperation): Boolean =
        selections[operation.ordinal].kernel != null && when (competitor) {
            RuntimeCompetitor.JvmScalar -> profile[operation].scalarToC
            RuntimeCompetitor.JvmVector -> profile[operation].simdToC
            RuntimeCompetitor.Native -> profile[operation].nativeToC
        } != WorkRule.Never

    fun explain(operation: DenseOperation, length: Int): String {
        val selected = selections[operation.ordinal].auto(competitor, length)
        if (selected != null) return describeNative(selected)
        val reason = if (operation.packed && !compatiblePacked) {
            "incompatible packed geometry"
        } else {
            "runtime policy or eligibility"
        }
        return "${runtime.explain(operation, length)}; $reason"
    }

    companion object {
        private val EMPTY_CONTEXT = NativeContext(0, 0, null, null)
    }
}

/** Composition occurs once. Hot paths return a prebound implementation, never an allocated plan. */
internal fun densePolicyEngine(
    runtime: KoblasEngine,
    native: KoblasEngine,
    competitor: RuntimeCompetitor,
    profile: DenseProfile = DenseProfiles.conservative,
): KoblasEngine {
    val dispatch = DenseDispatch(runtime, native, competitor, profile)
    val vector = if (DenseOperation.entries.any { !it.panel && !it.packed && dispatch.hasNativeChoice(it) }) {
        PolicyVectorKernels(runtime.vectorKernels, native.vectorKernels, dispatch)
    } else {
        runtime.vectorKernels
    }
    val panel = if (DenseOperation.entries.any { it.panel && dispatch.hasNativeChoice(it) }) {
        PolicyPanelKernels(runtime.panelKernels, native.panelKernels, dispatch)
    } else {
        runtime.panelKernels
    }
    val packed = if (DenseOperation.entries.any { it.packed && dispatch.hasNativeChoice(it) }) {
        PolicyPackedKernels(runtime.packedKernels, native.packedKernels, dispatch)
    } else {
        runtime.packedKernels
    }
    val indexed = if (competitor == RuntimeCompetitor.JvmVector) {
        runtime.indexedSparseKernels
    } else {
        native.indexedSparseKernels
    }
    val sparseName = if (competitor == RuntimeCompetitor.JvmVector) {
        runtime.sparseKernels.name
    } else {
        native.sparseKernels.name
    }
    return KoblasEngine(
        vector,
        panel,
        packed,
        SparseKernelAdapter(sparseName, vector, indexed),
        indexed,
        native.nativeVariant,
        dispatch,
    )
}

internal fun describeNative(kernel: NativeKernel): String =
    "native id=${kernel.id} variant=${kernel.variant} registerBits=${kernel.geometry.registerBits} " +
        "layouts=${kernel.geometry.leftLayout}/${kernel.geometry.rightLayout}/${kernel.geometry.outputLayout}"
