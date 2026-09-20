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
     * This is also what an ordinary call on this platform gets where the module resolved, so the two are
     * one engine rather than one engine named twice: the Level 2 panels, the Level 3 tiles and the diagonal
     * substitutions were held back from the default while no crossover had been established, and the final
     * calibration established one. Twenty-four whole operations chosen for where a vector body could lose —
     * orders below one execution group, products whose rows leave most of a tile empty, single right-hand
     * side solves — put this arm ahead of the portable matrix arithmetic at every one of them, over two
     * independent passes and a third at triple the warmup. What it does not settle is any machine but the
     * one it ran on, which is why the bodies below still fall back by shape rather than by policy.
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
