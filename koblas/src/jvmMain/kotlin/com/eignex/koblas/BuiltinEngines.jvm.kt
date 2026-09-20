package com.eignex.koblas

import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.SimdProductKernels
import com.eignex.koblas.dense.SimdTriangularKernels
import com.eignex.koblas.dense.SimdVectorKernels
import com.eignex.koblas.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.sparse.SimdIndexedSparseKernels
import com.eignex.koblas.sparse.SparseKernelAdapter

/** JVM built-in engines. */
@KoblasEngineApi
public actual object BuiltinEngines {
    /** Pure Kotlin scalar at every level, without resolving a host library. */
    @get:JvmStatic
    public actual val scalar: KoblasEngine by lazy {
        KoblasEngine(
            ScalarVectorKernels,
            SparseKernelAdapter("scalar", ScalarVectorKernels, ScalarIndexedSparseKernels),
            ScalarIndexedSparseKernels,
            null,
        )
    }

    /**
     * Every Vector API kernel this library owns, which is the arm a measurement of them names.
     *
     * The JVM default uses this same engine when the module is available. Matrix operations keep their
     * shared scheduling and use vector bodies where the window permits them, with portable fallbacks
     * reported by the operation's route. Local calibration supports this selection on the measured host;
     * it does not establish performance on every architecture.
     *
     * The availability test comes before [SimdPanelKernels] is named, and has to: resolving that object's
     * species is what initializing it does, so on a runtime without the module naming it at all is a linkage
     * error rather than a null.
     */
    @get:JvmStatic
    public actual val simd: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) {
            KoblasEngine(
                SimdVectorKernels,
                SparseKernelAdapter("simd-sparse", SimdVectorKernels, SimdIndexedSparseKernels),
                SimdIndexedSparseKernels,
                null,
                SimdPanelKernels,
                SimdProductKernels,
                SimdTriangularKernels,
            )
        } else {
            null
        }
    }
}

/** The Vector API composition where the module resolved, and the portable engine where it did not. */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = BuiltinEngines.simd ?: BuiltinEngines.scalar
