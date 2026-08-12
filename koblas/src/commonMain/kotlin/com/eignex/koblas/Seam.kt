package com.eignex.koblas

import kotlin.concurrent.Volatile

/**
 * One replaceable half of the compute seam. An install wins outright; a register is an offer ranked by
 * [Backend.priority].
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
     * The install override, else the strongest registration, else null. Null means use the compiled kernel,
     * since a reference implementation defined in terms of itself would recurse.
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
 * @param W the whole-backend interface, the pair of the two half interfaces.
 * @param default the portable implementation, used when nothing is installed or registered.
 * @param onFirstRead discovery to run exactly once before the first read.
 */
internal class ComposedSeam<W : Backend>(private val default: W, private val onFirstRead: Lazy<Unit>? = null) {
    @Volatile
    private var installed: W? = null

    @Volatile
    private var resolved: W? = null

    /** The override, else what the halves resolved to, else [default]. */
    val active: W
        get() {
            onFirstRead?.value
            return installed ?: resolved ?: default
        }

    /** Sets or (with null) clears the explicit override, which outranks the resolved halves. */
    fun install(backend: W?) {
        installed = backend
    }

    /** Records what the halves currently compose to. Null means no half is registered. */
    fun resolve(backend: W?) {
        resolved = backend
    }

    /** Clears the override and the composed result, leaving [default] active. */
    fun reset() {
        installed = null
        resolved = null
    }
}
