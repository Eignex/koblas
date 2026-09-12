package com.eignex.koblas

import com.eignex.koblas.dense.CPackedKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.CVectorKernels
import com.eignex.koblas.dense.DenseProfiles
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.RuntimeCompetitor
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdPackedKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.SimdVectorKernels
import com.eignex.koblas.dense.densePolicyEngine
import com.eignex.koblas.dense.describeSimdComponent
import com.eignex.koblas.internal.kernels.JvmCKernelBindings
import com.eignex.koblas.internal.kernels.NativeCatalog
import com.eignex.koblas.sparse.CIndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** JVM built-in engines. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
        )
    }

    /** Native C policy using the catalog's explicit ordinary implementation. */
    public actual val c: KoblasEngine? by lazy {
        DenseProfiles.preferredVariant(NativeCatalog.variants)?.let {
            densePolicyEngine(scalar, nativeEngine(it), RuntimeCompetitor.JvmScalar)
        }
    }

    /** Native variants available on this host. */
    public actual val nativeVariants: List<NativeVariant> get() = NativeCatalog.variants

    /** Binds exact dense native execution once, failing for unavailable variants. */
    public actual fun exactC(variant: NativeVariant): KoblasEngine = nativeEngine(variant)

    private fun nativeEngine(variant: NativeVariant): KoblasEngine {
        val bindings = JvmCKernelBindings(variant)
        val vector = CVectorKernels(bindings)
        val indexed = CIndexedSparseKernels(bindings)
        return KoblasEngine(
            vector,
            CPanelKernels(bindings),
            CPackedKernels(bindings),
            SparseKernelAdapter("c-scalar-indexed-policy", vector, indexed),
            indexed,
            variant,
        )
    }

    /** Vector API kernels when the incubator module resolved at startup. */
    @get:JvmStatic
    public actual val simd: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) {
            KoblasEngine(
                SimdVectorKernels,
                SimdPanelKernels,
                SimdPackedKernels,
                SparseKernelAdapter("simd-sparse", SimdVectorKernels, SimdIndexedSparseKernels),
                SimdIndexedSparseKernels,
                runtimeDescription = ::describeSimdComponent,
            )
        } else {
            null
        }
    }
}
