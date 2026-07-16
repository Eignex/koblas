package com.eignex.koblas

/** Non-JVM backend seam (native / JS / Wasm). A native LAPACK binding (cinterop) slots in here per
 *  target; until then these platforms use the portable [ReferenceLinearAlgebra]. */
actual fun platformLinearAlgebra(): LinearAlgebra? = null
