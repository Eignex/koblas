package com.eignex.koblas.dense.host.cblas

/**
 * The CBLAS enums and the LAPACKE layout macro by their ABI integer values. Both host bindings pass these
 * straight through as plain ints, so they are declared once rather than per binding.
 */
internal object Cblas {
    const val COL_MAJOR = 102
    const val NO_TRANS = 111
    const val TRANS = 112
    const val UPPER = 121
    const val LOWER = 122
    const val NON_UNIT = 131
    const val UNIT = 132
    const val LEFT = 141
    const val RIGHT = 142

    fun uploOf(lower: Boolean): Int = if (lower) LOWER else UPPER

    fun transOf(transpose: Boolean): Int = if (transpose) TRANS else NO_TRANS

    fun diagOf(unitDiag: Boolean): Int = if (unitDiag) UNIT else NON_UNIT

    fun sideOf(right: Boolean): Int = if (right) RIGHT else LEFT
}
