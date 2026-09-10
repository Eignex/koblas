package com.eignex.koblas

/** Short read-only identifier for the current process's complete built-in kernel engine. */
public val kernelBackend: String get() = koblas.name
