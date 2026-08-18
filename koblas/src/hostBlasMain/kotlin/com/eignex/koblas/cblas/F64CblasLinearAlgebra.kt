package com.eignex.koblas.cblas

import com.eignex.koblas.BackendNames
import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64LinearAlgebra
import com.eignex.koblas.dense.F64VectorKernels

/**
 * [F64LinearAlgebra] backed by the host's OpenBLAS through CBLAS and LAPACKE, resolved with `dlopen` on
 * first use. The two halves resolve independently, so [isAvailable] reports whether both did.
 */
public class F64CblasLinearAlgebra private constructor(
    private val blas: F64CblasBlas,
    private val lapack: F64CblasLapack,
) : F64LinearAlgebra,
    F64Blas by blas,
    F64Lapack by lapack {

    /** Resolves both halves, for a caller that wants to install the backend explicitly. */
    public constructor() : this(
        F64CblasBlas(requireNotNull(OpenBlasLoader.cblas) { NO_OPENBLAS }),
        F64CblasLapack(requireNotNull(OpenBlasLoader.lapacke) { NO_LAPACKE }, requireNotNull(OpenBlasLoader.cblas)),
    )

    override val name: String get() = BackendNames.CBLAS

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    override val isPortable: Boolean get() = false

    /** Both halves, since this type is the pair. Either half alone is reachable through the registry. */
    override val isAvailable: Boolean get() = blas.isAvailable && lapack.isAvailable

    /** The BLAS half's kernels, so both halves' inherited routines agree. */
    override val vectorKernels: F64VectorKernels get() = blas.vectorKernels

    /** Availability checks for the host CBLAS and LAPACKE. */
    public companion object {
        /** Whether the host provides both CBLAS and LAPACKE, so the full backend can be constructed. */
        public fun isAvailable(): Boolean = OpenBlasLoader.cblas != null && OpenBlasLoader.lapacke != null

        private const val NO_OPENBLAS =
            "OpenBLAS is not available on this host; koblas falls back to the reference backend"
        private const val NO_LAPACKE =
            "LAPACKE is not available on this host; koblas keeps its portable factorizations"
    }
}
