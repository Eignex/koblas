package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.VectorKernels

/**
 * What every backend reports about itself, whichever half of the seam it implements.
 *
 * A backend may implement any of the six halves — [Blas], [Lapack], [VectorKernels] and their sparse
 * counterparts — and [registerBackend] offers it as each one it implements. The halves are ranked and
 * selected independently, so a host that provides one library and not the other still accelerates what it
 * can.
 */
interface Backend {
    /** A short backend identifier for diagnostics (e.g. `"reference"`). */
    val name: String

    /**
     * Relative preference among simultaneously available backends: automatic selection through
     * [registerBackend] — JVM classpath discovery, native startup registration — picks the highest per half.
     *
     * The portable reference is 0 and every host binding koblas ships is [HOST_BACKEND_PRIORITY], on every
     * platform. A third-party backend that should win outright registers above that; one that should serve
     * only where no host library is installed registers between the two.
     */
    val priority: Int get() = 0
}

/**
 * The priority every host binding koblas ships registers at: OpenBLAS and UMFPACK, on the JVM and on native.
 *
 * One value rather than a scale, because these backends never compete with each other — they are ranked per
 * half, and no two of them implement the same half. What the number is for is third-party backends deciding
 * where they sit relative to "koblas found a host library", and that question has the same answer whichever
 * library and whichever platform is involved.
 *
 * ### The shape a host binding follows
 *
 * There are four of them — OpenBLAS and UMFPACK, each on the JVM and on the native targets — and they are
 * deliberately the same shape, so that a fifth is a transcription rather than a set of decisions:
 *
 * - **Resolved, never linked.** The library is opened by soname at runtime, so a binary carrying the binding
 *   still starts where the library is absent, and the portable implementation takes over.
 * - **A key symbol decides.** Opening is not enough: a named symbol has to resolve, which is what tells a
 *   library built without the expected variant (UMFPACK's `int64_t` family, an OpenBLAS without LAPACKE)
 *   apart from one that will work.
 * - **No computation runs at registration.** Resolution is the whole test. Running one would mean real work
 *   inside classpath discovery, which is where `Linker.downcallHandle` threw `StackOverflowError` on the JVM
 *   and reported it as a missing library. Third-party backends offered through discovery *are* probed, because
 *   nothing has checked them at all. The gap this leaves is an ABI mismatch: an OpenBLAS built with 64-bit
 *   integers exports the same names, so its symbols resolve and its results are wrong; detecting that
 *   wants `openblas_get_config`, which is a lookup rather than a computation, and is not done yet.
 * - **`available` is the question's name**, on every binding, with a second `…Available` beside it when a
 *   binding covers two independent libraries.
 * - **This priority.**
 *
 * What legitimately differs is when the symbols get bound, and only because the platforms differ in what that
 * costs. Creating a JVM downcall handle generates method-handle forms, so those are bound lazily — per symbol
 * for OpenBLAS's forty-odd, in one holder for UMFPACK's seven, which are all needed together anyway. A native
 * `dlsym` is a hash lookup, so the native bindings resolve everything eagerly when their loader initializes.
 */
const val HOST_BACKEND_PRIORITY = 100
