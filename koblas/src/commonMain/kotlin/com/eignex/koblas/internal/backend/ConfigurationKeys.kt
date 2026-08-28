package com.eignex.koblas.internal.backend

/**
 * The system properties and environment variables koblas reads. Each is an external identifier a caller
 * types on a command line, so they are collected here rather than spelled out at the one place that reads
 * them. Which platform honors which is up to that platform: there are no system properties outside the JVM.
 */
internal object ConfigurationKeys {
    /** Pins one semantic backend role to one [BackendNames] value instead of taking the highest offer. */
    val BACKENDS: Map<BackendSlot, BackendSelectionKeys> = mapOf(
        BackendSlot.F64Kernels to BackendSelectionKeys(
            "koblas.backend.dense.kernels",
            "KOBLAS_DENSE_KERNELS_BACKEND",
        ),
        BackendSlot.F64Blas to BackendSelectionKeys(
            "koblas.backend.dense.blas",
            "KOBLAS_DENSE_BLAS_BACKEND",
        ),
        BackendSlot.F64Decompositions to BackendSelectionKeys(
            "koblas.backend.dense.decompositions",
            "KOBLAS_DENSE_DECOMPOSITIONS_BACKEND",
        ),
        BackendSlot.F64SparseKernels to BackendSelectionKeys(
            "koblas.backend.sparse.kernels",
            "KOBLAS_SPARSE_KERNELS_BACKEND",
        ),
        BackendSlot.F64SparseBlas to BackendSelectionKeys(
            "koblas.backend.sparse.blas",
            "KOBLAS_SPARSE_BLAS_BACKEND",
        ),
        BackendSlot.F64GeneralSparseLu to BackendSelectionKeys(
            "koblas.backend.sparse.general.lu",
            "KOBLAS_SPARSE_GENERAL_LU_BACKEND",
        ),
        BackendSlot.F64RepeatedSparseLu to BackendSelectionKeys(
            "koblas.backend.sparse.repeated.lu",
            "KOBLAS_SPARSE_REPEATED_LU_BACKEND",
        ),
        BackendSlot.F64SparseCholesky to BackendSelectionKeys(
            "koblas.backend.sparse.cholesky",
            "KOBLAS_SPARSE_CHOLESKY_BACKEND",
        ),
        BackendSlot.F64SparseLdl to BackendSelectionKeys(
            "koblas.backend.sparse.ldl",
            "KOBLAS_SPARSE_LDL_BACKEND",
        ),
        BackendSlot.F64SparseQr to BackendSelectionKeys(
            "koblas.backend.sparse.qr",
            "KOBLAS_SPARSE_QR_BACKEND",
        ),
        BackendSlot.F64BasisFactorizations to BackendSelectionKeys(
            "koblas.backend.basis.factorizations",
            "KOBLAS_BASIS_FACTORIZATIONS_BACKEND",
        ),
        BackendSlot.F64BasisSolvers to BackendSelectionKeys(
            "koblas.backend.basis.solvers",
            "KOBLAS_BASIS_SOLVERS_BACKEND",
        ),
    )

    /** An absolute path to the library exporting `cblas_*`, overriding the deployment lookup chain. */
    val CBLAS_PATH = LibraryPathKeys("koblas.cblas.path", "KOBLAS_CBLAS_PATH")

    /** An absolute path to the library exporting `LAPACKE_*`, for a host that keeps it outside its CBLAS. */
    val LAPACKE_PATH = LibraryPathKeys("koblas.lapacke.path", "KOBLAS_LAPACKE_PATH")

    /** An absolute path to the host KLU. */
    val KLU_PATH = LibraryPathKeys("koblas.klu.path", "KOBLAS_KLU_PATH")

    /** An absolute path to the host UMFPACK. */
    val UMFPACK_PATH = LibraryPathKeys("koblas.umfpack.path", "KOBLAS_UMFPACK_PATH")

    /** An absolute path to the host CHOLMOD. */
    val CHOLMOD_PATH = LibraryPathKeys("koblas.cholmod.path", "KOBLAS_CHOLMOD_PATH")

    /** An absolute path to a BASICLU exporting koblas's bridge entry points, ahead of the bundled build. */
    val BASICLU_PATH = LibraryPathKeys("koblas.basiclu.path", "KOBLAS_BASICLU_PATH")

    /** An absolute path to a build of koblas's HFactor bridge, ahead of the bundled one. */
    val HFACTOR_PATH = LibraryPathKeys("koblas.hfactor.path", "KOBLAS_HFACTOR_PATH")
}

/** The system property and the environment variable a deployment can point one library path at. */
internal class LibraryPathKeys(val property: String, val environment: String)

/** The system property and environment variable that pin one semantic backend role. */
internal class BackendSelectionKeys(val property: String, val environment: String)

/**
 * The backend pin a deployment asked for, [property] ahead of [environment]. Blank counts as unset, since a
 * variable exported empty is a deployment that meant to clear the pin rather than one asking for a backend
 * no name matches.
 */
internal fun pinnedBackend(property: String?, environment: String?): String? =
    (property ?: environment)?.trim()?.takeIf { it.isNotEmpty() }
