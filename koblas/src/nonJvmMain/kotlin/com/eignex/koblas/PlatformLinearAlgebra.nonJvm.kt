package com.eignex.koblas

/** Non-JVM backend seam (native / JS / Wasm). These platforms have no runtime discovery, so they
 *  default to the portable [ReferenceLinearAlgebra]; a backend artifact such as koblas-cblas is
 *  activated explicitly via [installLinearAlgebra]. */
actual fun platformLinearAlgebra(): LinearAlgebra? = null
