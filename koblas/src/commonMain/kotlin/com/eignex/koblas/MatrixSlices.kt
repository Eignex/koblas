@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.sparse.*

/**
 * Fresh transposed matrix, with the active backend ([koblas]). For products, prefer the transpose flags on
 * gemv and gemm, which read the original storage without copying. See [KoblasEngine.transpose].
 */
public fun DenseMatrix.transpose(): DenseMatrix = koblas.transpose(this)

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
 * zeros survive, since the transpose is structural rather than arithmetic.
 */
public fun SparseMatrix.transpose(): SparseMatrix = koblas.transpose(this)
