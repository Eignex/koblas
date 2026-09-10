package com.eignex.koblas.sparse.host.hfactor

/** HFactor's basis-update representations. */
public enum class HfactorUpdateMethod(internal val nativeValue: Int) {
    /** Forrest-Tomlin updates: the usual sparse-simplex default. */
    FORREST_TOMLIN(1),

    /** Product-form updates. */
    PRODUCT_FORM(2),

    /** Middle product-form updates. */
    MIDDLE_PRODUCT_FORM(3),

    /** Alternate product-form updates. */
    ALTERNATE_PRODUCT_FORM(4),
}

/** Policy for one HFactor solver factory. */
public data class HfactorConfig(
    /** An absolute path to the library, or the platform lookup chain when null. */
    val libraryPath: String? = null,
    /** Whether to scale rows before factorizing and undo it in the solves. */
    val equilibrate: Boolean = false,
    /** Markowitz pivot acceptance as a fraction of the largest entry in its column. */
    val pivotThreshold: Double = 0.1,
    /** Smallest acceptable absolute pivot. */
    val pivotTolerance: Double = 1e-10,
    /** Factor update representation retained between reinversions. */
    val updateMethod: HfactorUpdateMethod = HfactorUpdateMethod.FORREST_TOMLIN,
) {
    init {
        require(pivotThreshold in 8e-4..0.5 && pivotThreshold.isFinite()) {
            "pivotThreshold must be finite and in [8e-4, 0.5]: $pivotThreshold"
        }
        require(pivotTolerance in 0.0..1.0 && pivotTolerance.isFinite()) {
            "pivotTolerance must be finite and in [0, 1]: $pivotTolerance"
        }
    }
}

/**
 * Names the platform loader looks for. HFactor is a C++ class inside HiGHS rather than a library with an
 * API of its own, so what is looked for is koblas's own build of it and not a host HiGHS: no distribution
 * ships these entry points.
 */
internal val HFACTOR_SONAMES: List<String> = listOf(
    "libkoblas_hfactor.so.1",
    "libkoblas_hfactor.so",
    "libkoblas_hfactor.1.dylib",
    "libkoblas_hfactor.dylib",
)
