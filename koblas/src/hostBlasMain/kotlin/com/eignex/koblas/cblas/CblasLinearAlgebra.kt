package com.eignex.koblas.cblas

import com.eignex.koblas.Blas
import com.eignex.koblas.Lapack
import com.eignex.koblas.LinearAlgebra

/**
 * [LinearAlgebra] backed by the host's OpenBLAS through its C interfaces (CBLAS and LAPACKE), for
 * the Linux and macOS native targets. Nothing is linked: the libraries are resolved with `dlopen`
 * at program start, so the dependency is optional at runtime — `libopenblas` plus `liblapacke` on
 * Debian/Ubuntu, `brew install openblas` on macOS. When they are present koblas installs this backend
 * eagerly before `main`; when they are missing the program still runs on
 * [com.eignex.koblas.ReferenceLinearAlgebra], and constructing this class throws. [isAvailable] and
 * [isBlasAvailable] report which case the host is, since the two halves resolve independently.
 *
 * Koblas storage is column-major, the order LAPACK defines, so buffers cross the FFI
 * boundary without repacking. Semantics match [com.eignex.koblas.ReferenceLinearAlgebra] exactly as
 * specified by the [LinearAlgebra] contract: `beta == 0` overwrites without reading, `alpha == 0`
 * reduces to the `beta` scale, [syrk] produces the full, exactly symmetric result by default, and
 * the factorizations use the shared packed formats so they interchange between backends.
 *
 * OpenBLAS runs single-threaded by default here, which is the faster configuration at koblas
 * workload sizes; set the `OPENBLAS_NUM_THREADS` environment variable to opt into its threading.
 */
class CblasLinearAlgebra private constructor(private val blas: CblasBlas, private val lapack: CblasLapack) :
    LinearAlgebra,
    Blas by blas,
    Lapack by lapack {

    /** Both halves of the host library, for a caller that wants to install it explicitly. */
    constructor() : this(
        CblasBlas(requireNotNull(OpenBlasLoader.cblas) { NO_OPENBLAS }),
        CblasLapack(requireNotNull(OpenBlasLoader.lapacke) { NO_LAPACKE }, requireNotNull(OpenBlasLoader.cblas)),
    )

    override val name: String get() = "cblas"

    /** Above the reference (0). */
    override val priority: Int get() = 90

    /** Host-availability checks for the two halves. */
    companion object {
        /** Whether the host provides both CBLAS and LAPACKE, so the full backend can be constructed. */
        fun isAvailable(): Boolean = OpenBlasLoader.cblas != null && OpenBlasLoader.lapacke != null

        /** Whether the host provides CBLAS, which is all the [Blas] half needs. */
        fun isBlasAvailable(): Boolean = OpenBlasLoader.cblas != null

        private const val NO_OPENBLAS =
            "OpenBLAS is not available on this host; koblas falls back to the reference backend"
        private const val NO_LAPACKE =
            "LAPACKE is not available on this host; koblas keeps its portable factorizations"
    }
}
