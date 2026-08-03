package com.eignex.koblas

/**
 * Identifies the runtime math backend powering the SIMD-like primitives.
 *
 * `scalar` is the built-in loops; `scalar+host` means a [Level1] backend is active, so runs of at least
 * [HOST_LEVEL1_MIN_LENGTH] elements go to a host BLAS while shorter ones stay on the loops. The read is
 * dynamic because that install happens at program start, and a run that silently kept the scalar kernels
 * is exactly the failure this is here to report.
 */
public actual val mathBackend: String
    get() = if (activeLevel1 != null) "scalar+host" else "scalar"
