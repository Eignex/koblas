package com.eignex.koblas.cblas

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.LinearAlgebra
import com.eignex.koblas.dense.VectorKernels

/**
 * [LinearAlgebra] backed by the host's OpenBLAS through CBLAS and LAPACKE, resolved with `dlopen` on
 * first use. The two halves resolve independently, so check [isAvailable] and [isBlasAvailable].
 */
public class CblasLinearAlgebra private constructor(private val blas: CblasBlas, private val lapack: CblasLapack) :
    LinearAlgebra,
    Blas by blas,
    Lapack by lapack {

    /** Resolves both halves, for a caller that wants to install the backend explicitly. */
    public constructor() : this(
        CblasBlas(requireNotNull(OpenBlasLoader.cblas) { NO_OPENBLAS }),
        CblasLapack(requireNotNull(OpenBlasLoader.lapacke) { NO_LAPACKE }, requireNotNull(OpenBlasLoader.cblas)),
    )

    override val name: String get() = "cblas"

    override val priority: Int get() = HOST_BACKEND_PRIORITY

    /** The BLAS half's kernels, so both halves' inherited routines agree on where their inner loops go. */
    override val vectorKernels: VectorKernels get() = blas.vectorKernels

    /** Availability checks for the host CBLAS and LAPACKE. */
    public companion object {
        /** Whether the host provides both CBLAS and LAPACKE, so the full backend can be constructed. */
        public fun isAvailable(): Boolean = OpenBlasLoader.cblas != null && OpenBlasLoader.lapacke != null

        /** Whether the host provides CBLAS, which is all the [Blas] half needs. */
        public fun isBlasAvailable(): Boolean = OpenBlasLoader.cblas != null

        private const val NO_OPENBLAS =
            "OpenBLAS is not available on this host; koblas falls back to the reference backend"
        private const val NO_LAPACKE =
            "LAPACKE is not available on this host; koblas keeps its portable factorizations"
    }
}
