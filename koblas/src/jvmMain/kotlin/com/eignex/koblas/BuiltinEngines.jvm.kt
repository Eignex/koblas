package com.eignex.koblas

import com.eignex.koblas.dense.CKernels
import com.eignex.koblas.dense.CPackedKernels
import com.eignex.koblas.dense.CPanelKernels
import com.eignex.koblas.dense.PortablePackedKernels
import com.eignex.koblas.dense.ScalarKernels
import com.eignex.koblas.dense.ScalarPanelKernels
import com.eignex.koblas.dense.SimdKernels
import com.eignex.koblas.dense.SimdPackedKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.sparse.CSparseKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.ScalarSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels
import com.eignex.koblas.sparse.SimdSparseKernels

/** JVM built-in engines. */
public actual object BuiltinEngines {
    /** Pure Kotlin scalar dense kernels and reference sparse kernels. */
    public actual val scalar: KoblasContext by lazy {
        KoblasContext(
            ScalarKernels,
            ScalarPanelKernels,
            PortablePackedKernels,
            ScalarSparseKernels,
            ScalarIndexedSparseKernels,
        )
    }

    /** Bundled C kernels when the native library loaded successfully. */
    public actual val c: KoblasContext? by lazy {
        if (CKernels.isAvailable) {
            KoblasContext(
                CKernels,
                CPanelKernels,
                CPackedKernels,
                CSparseKernels,
                ScalarIndexedSparseKernels,
            )
        } else {
            null
        }
    }

    /** Vector API kernels when the incubator module resolved at startup. */
    public actual val simd: KoblasContext? by lazy {
        if (SimdKernels.isAvailable) {
            KoblasContext(
                SimdKernels,
                SimdPanelKernels,
                SimdPackedKernels,
                SimdSparseKernels,
                SimdIndexedSparseKernels,
            )
        } else {
            null
        }
    }
}
