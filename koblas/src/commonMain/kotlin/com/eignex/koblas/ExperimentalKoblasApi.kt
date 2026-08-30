package com.eignex.koblas

/** Marks APIs that expose implementation choices which may evolve between releases. */
@RequiresOptIn(
    message = "This API exposes experimental koblas implementation choices",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
public annotation class ExperimentalKoblasApi
