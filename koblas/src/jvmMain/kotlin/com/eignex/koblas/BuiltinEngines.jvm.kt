package com.eignex.koblas

import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.DenseProductKernels
import com.eignex.koblas.dense.PortablePanelKernels
import com.eignex.koblas.dense.PortableProductKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.dense.SimdPanelKernels
import com.eignex.koblas.dense.SimdProductKernels
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
     * The availability test comes before [SimdPanelKernels] is named, and has to: resolving that object's
     * species is what initializing it does, so on a runtime without the module naming it at all is a linkage
     * error rather than a null.
     */
    @get:JvmStatic
    public actual val simd: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) simdEngine(SimdPanelKernels, SimdProductKernels) else null
    }

    /**
     * What this platform selects, which is [simd]'s Level 1 with the portable panels and product tiles.
     *
     * The Level 1 kernels are measured and have been the default since they were. The Level 2 panels and the
     * Level 3 tiles are not: they are correct, allocation-free and locally ahead over most shapes, and no
     * crossover has been established across machines. Activating one is its own decision with its own
     * evidence, so until then an ordinary call keeps the portable matrix arithmetic and [simd] is where the
     * vector one is measured.
     */
    internal val defaultComposition: KoblasEngine? by lazy {
        if (SimdVectorKernels.isAvailable) simdEngine(PortablePanelKernels, PortableProductKernels) else null
    }

    /** The Vector API Level 1 composition, with whichever matrix arithmetic a caller above has decided on. */
    private fun simdEngine(panels: DensePanelKernels, products: DenseProductKernels): KoblasEngine = KoblasEngine(
        SimdVectorKernels,
        SparseKernelAdapter("simd-sparse", SimdVectorKernels, SimdIndexedSparseKernels),
        SimdIndexedSparseKernels,
        null,
        panels,
        products,
    )
}

/** The default composition where the module resolved, and the portable engine where it did not. */
@OptIn(KoblasEngineApi::class)
internal actual fun platformEngine(): KoblasEngine = BuiltinEngines.defaultComposition ?: BuiltinEngines.scalar
