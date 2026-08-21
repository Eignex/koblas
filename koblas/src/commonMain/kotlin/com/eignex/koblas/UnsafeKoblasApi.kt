package com.eignex.koblas

/**
 * Marks direct access to live structural or factorization storage. Mutating these buffers can invalidate
 * later operations; access is intended for backend interoperability and specialized kernels.
 */
@RequiresOptIn(
    message = "This is live internal storage; mutating it can invalidate the owning object",
    level = RequiresOptIn.Level.ERROR,
)
@Retention(AnnotationRetention.BINARY)
@Target(AnnotationTarget.CONSTRUCTOR, AnnotationTarget.PROPERTY)
public annotation class UnsafeKoblasApi
