package com.eignex.koblas

/**
 * Allows exact built-in engine selection and shape diagnostics for benchmarks and conformance tests.
 *
 * Application convenience operations use [koblas], the platform default. Selecting a [BuiltinEngines]
 * instance does not change that default; calls through the instance use its own kernels and fallbacks.
 * Application code can explicitly select portable Kotlin with [KoblasEngine.portable] without this opt-in.
 */
@RequiresOptIn(
    message = "Exact engines are for tests; production code uses koblas or KoblasEngine.portable()",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.PROPERTY, AnnotationTarget.FUNCTION)
public annotation class KoblasEngineApi
