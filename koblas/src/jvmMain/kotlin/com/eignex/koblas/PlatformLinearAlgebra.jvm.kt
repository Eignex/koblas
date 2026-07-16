package com.eignex.koblas

/** JVM backend seam. A native LAPACK binding (FFM → OpenBLAS/MKL, or a JVM LAPACK) slots in here; until
 *  then the JVM uses the portable [ReferenceLinearAlgebra]. */
actual fun platformLinearAlgebra(): LinearAlgebra? = null
