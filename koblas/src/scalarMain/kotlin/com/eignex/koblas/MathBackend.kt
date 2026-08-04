package com.eignex.koblas

import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.dense.vectorKernelSeam

/**
 * Identifies the runtime math backend powering the SIMD-like primitives.
 *
 * `scalar` is the built-in loops; `scalar+host` means a [VectorKernels] backend is active, so runs of at
 * least [DispatchThresholds.level1] elements go to a host BLAS while shorter ones stay on the loops. The
 * read is dynamic because that registration happens at program start, and a run that silently kept the
 * scalar kernels is exactly the failure this is here to report.
 */
public actual val mathBackend: String
    get() = if (vectorKernelSeam.active != null) "scalar+host" else "scalar"
