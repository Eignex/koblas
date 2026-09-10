package com.eignex.koblas

import com.eignex.koblas.dense.CPackedKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.CVectorKernels
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdPackedKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.SimdVectorKernels
import com.eignex.koblas.sparse.CIndexedSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** JVM built-in engines. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasContext by lazy {
        KoblasContext(
            ScalarVectorKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
        )
    }

    /** Bundled C kernels when the native library loaded successfully. */
    public actual val c: KoblasContext? by lazy {
        if (CVectorKernels.isAvailable) {
            KoblasContext(
                CVectorKernels,
                CPanelKernels,
                CPackedKernels,
                SparseKernelAdapter("c-sparse", CVectorKernels, CIndexedSparseKernels),
                CIndexedSparseKernels,
            )
        } else {
            null
        }
    }

    /** Vector API kernels when the incubator module resolved at startup. */
    public actual val simd: KoblasContext? by lazy {
        if (SimdVectorKernels.isAvailable) {
            KoblasContext(
                SimdVectorKernels,
                SimdPanelKernels,
                SimdPackedKernels,
                SparseKernelAdapter("simd-sparse", SimdVectorKernels, SimdIndexedSparseKernels),
                SimdIndexedSparseKernels,
            )
        } else {
            null
        }
    }
}
