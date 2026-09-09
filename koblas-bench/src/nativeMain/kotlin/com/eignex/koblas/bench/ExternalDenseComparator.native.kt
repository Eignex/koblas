package com.eignex.koblas.bench

import com.eignex.koblas.*

@OptIn(ExperimentalKoblasApi::class)
internal actual fun explicitBuiltInContext(): KoblasContext {
    val provider = BuiltinKernels.c ?: BuiltinKernels.scalar
    return ContextBuilder().withBuiltinKernels(provider).resolve()
}

// The benchmark-owned native cinterop is added per target once that target has an OpenBLAS SDK available.
// Returning unavailable is deliberate: it cannot silently become a koblas host call and be credited as an
// independent comparison. The JVM comparator remains the permanent x86-64 measurement arm.
