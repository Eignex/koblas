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
 * One value rather than a scale: these four never compete, since no two implement the same half. It exists so
 * a third-party backend can decide where it sits relative to "koblas found a host library".
 *
 * All four follow one shape, so a fifth is a transcription rather than a set of decisions. The library is
 * resolved by soname at runtime and never linked; a named symbol has to resolve, not just the library open;
 * no computation runs at registration, since that work inside discovery is where `Linker.downcallHandle` threw
 * `StackOverflowError` and was reported as a missing library; `available` is the name of the question, with a
 * second `…Available` where a binding covers two libraries. Third-party backends *are* probed, having been
 * checked by nothing. The gap that leaves — an ILP64 OpenBLAS exports the same names and computes wrong
 * answers — is caught by reading `openblas_get_config`, a lookup rather than a computation.
 *
 * Only the timing of symbol binding differs, because the platforms differ in what it costs: JVM downcall
 * handles generate method-handle forms and are bound lazily, while a native `dlsym` is a hash lookup and the
 * native loaders resolve everything at once.
 */
const val HOST_BACKEND_PRIORITY = 100
