package com.eignex.koblas

import kotlin.concurrent.Volatile

// The selection machinery behind every replaceable half of koblas, in one place.
//
// koblas has six of them - dense `Blas`, `Lapack` and `VectorKernels`, and the three sparse counterparts -
// and each would otherwise carry its own copy of the same state machine: two fields, a priority comparison, a
// recompose, a test hook. Six copies of one mechanism is six chances for them to drift apart, and they had
// already started to: only one of the reset hooks cleared its install override.
//
// The two classes below are the mechanism. What stays hand-written per side is the part that genuinely
// differs - how two halves compose into a whole, which needs `by` delegation over the specific interfaces
// and cannot be expressed generically.

/**
 * One replaceable half of the compute seam: which backend is active, and how it got there.
 *
 * Two ways in, and the precedence between them is the whole contract. An `install` is an explicit choice by
 * the embedding program and wins outright; a `register` is an offer, ranked against other offers by
 * [Backend.priority], so the strongest candidate wins and a later weaker one does not displace it. With
 * neither, [active] is null and the caller decides what that means.
 *
 * [active] is deliberately nullable rather than defaulting to a reference implementation, because the two
 * kinds of half need different things from "nothing selected". The matrix seams fall back to an object, so
 * they read `active ?: reference`. The dense vector seam cannot: its fallback is a compiled-in primitive
 * function, and a reference `dot` implemented in terms of `dot` would be recursion rather than a fallback.
 * There, null *is* the answer - use the compiled kernel - and `Primitives.kt` tests this field before it
 * even looks at a length.
 *
 * @param T the interface this half implements.
 * @param onChange run after every change, for a seam whose composed whole needs rebuilding.
 */
internal class Seam<T : Backend>(private val onChange: () -> Unit = {}) {
    @Volatile
    private var installed: T? = null

    @Volatile
    private var registered: T? = null

    /**
     * The selected backend: the install override, else the strongest registration, else null.
     *
     * A plain field rather than a computed property because the dense vector seam is read by every `dot` in
     * the library. One volatile read is the floor for something another thread can change, and this is it.
     */
    @Volatile
    var active: T? = null
        private set

    /** Sets or (with null) clears the explicit override, which outranks any registration. */
    fun install(backend: T?) {
        installed = backend
        recompose()
    }

    /** Offers [backend], which becomes active only if it outranks the strongest previous offer. */
    fun register(backend: T) {
        val current = registered
        if (current == null || backend.priority > current.priority) {
            registered = backend
            recompose()
        }
    }

    /** Clears both the override and the registration, leaving [active] null. */
    fun reset() {
        installed = null
        registered = null
        recompose()
    }

    private fun recompose() {
        active = installed ?: registered
        onChange()
    }
}

/**
 * A whole backend assembled from two [Seam] halves, with a portable default and an override of its own.
 *
 * Separate from [Seam] because the halves and the whole are selected differently. A half is offered and
 * ranked; the whole is either explicitly installed, or [resolve]d from whichever halves won, or the
 * portable [default] - so unlike [Seam.active] this never returns null, and there is always something to
 * call.
 *
 * What the halves compose to is cached in a field rather than composed on demand: [active] is read on every
 * dispatched call, and composing per call would allocate per call.
 *
 * @param W the whole-backend interface, the pair of the two half interfaces.
 * @param default the portable implementation, used when nothing is installed or registered.
 * @param onFirstRead discovery to run exactly once before the first read - the dense side scans for host
 *   backends here. Reading it inside the getter rather than at construction is what makes a test that
 *   clears the registry stay cleared, instead of having discovery silently repopulate it.
 */
internal class ComposedSeam<W : Backend>(private val default: W, private val onFirstRead: Lazy<Unit>? = null) {
    @Volatile
    private var installed: W? = null

    @Volatile
    private var resolved: W? = null

    /** The active backend: the override, else what the halves resolved to, else [default]. */
    val active: W
        get() {
            onFirstRead?.value
            return installed ?: resolved ?: default
        }

    /** Sets or (with null) clears the explicit override, which outranks the resolved halves. */
    fun install(backend: W?) {
        installed = backend
    }

    /** Records what the halves currently compose to; null means no half is registered. */
    fun resolve(backend: W?) {
        resolved = backend
    }

    /** Clears the override and the composed result, leaving [default] active. */
    fun reset() {
        installed = null
        resolved = null
    }
}
