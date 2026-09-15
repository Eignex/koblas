package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines

/** Exact scalar oracle shared by sparse conformance tests, over sparse Level 1 and the indexed kernels. */
internal object ReferenceSparseBlas : SparseKernels by BuiltinEngines.scalar.sparseKernels
