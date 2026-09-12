package com.eignex.koblas

import com.eignex.koblas.dense.DenseProfiles
import com.eignex.koblas.dense.NativeCPackedKernels
import com.eignex.koblas.dense.NativeCPanelKernels
import com.eignex.koblas.dense.NativeCVectorKernels
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.RuntimeCompetitor
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.densePolicyEngine
import com.eignex.koblas.internal.kernels.NativeCKernelBindings
import com.eignex.koblas.internal.kernels.NativeCatalog
import com.eignex.koblas.sparse.NativeCIndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** Kotlin/Native built-in engines. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasEngine =
        KoblasEngine(
            ScalarVectorKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
        )

    /** Native C policy using the catalog's explicit ordinary implementation. */
    public actual val c: KoblasEngine? by lazy {
        DenseProfiles.preferredVariant(NativeCatalog.variants)?.let {
            densePolicyEngine(scalar, nativeEngine(it), RuntimeCompetitor.Native)
        }
    }

    /** Native variants available on this host. */
    public actual val nativeVariants: List<NativeVariant> get() = NativeCatalog.variants

    /** Binds exact dense native execution once, failing for unavailable variants. */
    public actual fun exactC(variant: NativeVariant): KoblasEngine = nativeEngine(variant)

    private fun nativeEngine(variant: NativeVariant): KoblasEngine {
        val bindings = NativeCKernelBindings(variant)
        val vector = NativeCVectorKernels(bindings)
        val indexed = NativeCIndexedSparseKernels(bindings)
        return KoblasEngine(
            vector,
            NativeCPanelKernels(bindings),
            NativeCPackedKernels(bindings),
            SparseKernelAdapter("c-scalar-indexed-policy", vector, indexed),
            indexed,
            variant,
        )
    }

    /** SIMD is unavailable as a distinct Native engine. */
    public actual val simd: KoblasEngine? = null
}
