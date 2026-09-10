package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines

/** Exact scalar oracle shared by sparse conformance tests. */
internal object ReferenceSparseBlas :
    SparseBlas by BuiltinEngines.scalar,
    SparseKernels by BuiltinEngines.scalar.sparseKernels
