package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines

/**
 * Exact scalar oracle shared by sparse conformance tests.
 *
 * Covers the sparse Level 1 and indexed primitives only. It delegated to the sparse Level 2-3 seam as well
 * while one existed; the primitives are what remains, and they are what a platform kernel is compared against.
 */
internal object ReferenceSparseBlas : SparseKernels by BuiltinEngines.scalar.sparseKernels
