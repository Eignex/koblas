package com.eignex.koblas

/**
 * Reusable scratch for operations that would otherwise allocate their temporaries on every call.
 *
 * It is a set of named buffers, not a general allocator: each internal use site owns a distinct slot, so
 * one workspace can serve nested operations (a condition estimate driving triangular solves, say)
 * without two of them writing the same array. A buffer is cached per slot and reused whenever the
 * requested size matches, which is the case in the loops this exists for — a simplex iteration or a
 * filter update asks for the same shapes every time, so after the first call there are no allocations at
 * all. A changed size simply reallocates that slot.
 *
 * Buffer contents are undefined on entry: an operation that needs zeros clears what it uses.
 *
 * **Not thread-safe, deliberately.** A workspace is caller-owned state, so give each solver instance (or
 * each thread) its own; sharing one across concurrent operations corrupts results. Passing none keeps the
 * allocating behaviour, which is always correct.
 *
 * ```
 * val ws = Workspace()
 * repeat(iterations) {
 *     koblas.solveInto(lu, b, x)          // writes x, allocates nothing
 *     val rc = koblas.rcond(lu, anorm, ws) // reuses ws's buffers, allocates nothing
 * }
 * ```
 */
class Workspace {
    private val doubles = arrayOfNulls<DoubleArray>(SLOT_COUNT)
    private val ints = arrayOfNulls<IntArray>(SLOT_COUNT)

    /** The buffer for [slot], sized exactly [size]. Contents are undefined. */
    internal fun doubles(slot: Int, size: Int): DoubleArray {
        val cached = doubles[slot]
        if (cached != null && cached.size == size) return cached
        val fresh = DoubleArray(size)
        doubles[slot] = fresh
        return fresh
    }

    /** The int buffer for [slot], sized exactly [size]. Contents are undefined. */
    internal fun ints(slot: Int, size: Int): IntArray {
        val cached = ints[slot]
        if (cached != null && cached.size == size) return cached
        val fresh = IntArray(size)
        ints[slot] = fresh
        return fresh
    }

    /**
     * A buffer reserved for [LinearAlgebra] backends outside this module, sized exactly [size]. Koblas's
     * own routines never touch this slot, so a backend can stage in it without coordinating with them —
     * a backend's staging is always a leaf within one call, so one buffer is enough.
     */
    fun backendVector(size: Int): DoubleArray = doubles(BACKEND, size)

    /** Releases the cached buffers. Only useful when a workspace outlives the shapes it served. */
    fun clear() {
        doubles.fill(null)
        ints.fill(null)
    }

    internal companion object {
        // One slot per use site. Operations that call each other must not share a slot; the condition
        // estimator calls the triangular solves, which take a destination and need no scratch of their own.
        const val RCOND_X = 0
        const val RCOND_Y = 1
        const val RCOND_SIGNS = 2
        const val RCOND_PROBE = 3
        const val SOLVE_STAGE = 4
        const val SPARSE_WORK = 5
        const val SPARSE_WORK_2 = 6
        const val ETA_WORK = 7
        const val BACKEND = 8
        const val SLOT_COUNT = 9
    }
}
