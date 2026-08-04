package com.eignex.koblas.sparse

/**
 * Mutable open-addressing `Int → Double` hash map specialised to avoid the key/value autoboxing a
 * stdlib `HashMap<Int, Double>` pays on every `put`/`get`. Backs the per-row working state of
 * [SparseLu.factorize], whose elimination inner loops look up, insert, and remove entries per
 * fill-in update.
 *
 * Linear-probed, capacity a power of two, load factor kept ≤ 0.5. Occupancy is tracked by a parallel
 * `used` bitmap rather than a sentinel key, so **any** `Int` key is storable. Removal uses
 * backward-shift deletion (Knuth 6.4 algorithm R), so the table holds no tombstones.
 *
 * Not thread-safe. Iteration order is unspecified (hash order).
 */
internal class MutableIntDoubleMap(initialCapacity: Int = 8) {
    private var keys: IntArray
    private var values: DoubleArray
    private var used: BooleanArray
    private var mask: Int

    /** Number of entries currently stored. */
    var size: Int = 0
        private set

    init {
        val cap = openAddressingCapacity(initialCapacity)
        keys = IntArray(cap)
        values = DoubleArray(cap)
        used = BooleanArray(cap)
        mask = cap - 1
    }

    /** Value for [key], or [default] if absent. */
    fun getOrDefault(key: Int, default: Double): Double {
        val i = slotOf(key)
        return if (i >= 0) values[i] else default
    }

    /** Slot index of [key], or `-1` if absent; read the value with [valueAt]. Slots are invalidated
     *  by any mutation. */
    fun slotOf(key: Int): Int {
        var i = mixIntKey(key) and mask
        while (used[i]) {
            if (keys[i] == key) return i
            i = (i + 1) and mask
        }
        return -1
    }

    /** Value stored at [slot], which must come from a matching [slotOf] call. */
    fun valueAt(slot: Int): Double = values[slot]

    /** Insert or overwrite [key] → [value]. */
    fun put(key: Int, value: Double) {
        var i = mixIntKey(key) and mask
        while (used[i]) {
            if (keys[i] == key) {
                values[i] = value
                return
            }
            i = (i + 1) and mask
        }
        used[i] = true
        keys[i] = key
        values[i] = value
        size++
        if (size * 2 > keys.size) grow()
    }

    /** Remove [key]; returns true if it was present. Backward-shift keeps the table tombstone-free. */
    fun remove(key: Int): Boolean {
        var i = mixIntKey(key) and mask
        while (used[i]) {
            if (keys[i] == key) {
                deleteSlot(i)
                size--
                return true
            }
            i = (i + 1) and mask
        }
        return false
    }

    /** Multiply every stored value by [factor] in place. */
    fun scaleValues(factor: Double) {
        for (i in used.indices) if (used[i]) values[i] *= factor
    }

    /** Invoke [action] for each (key, value) entry. Iteration order is unspecified. The map must not
     *  be structurally modified during iteration (in-place value writes are fine). */
    inline fun forEach(action: (key: Int, value: Double) -> Unit) {
        val k = keysInternal
        val v = valuesInternal
        val u = usedInternal
        for (i in u.indices) if (u[i]) action(k[i], v[i])
    }

    @PublishedApi internal val keysInternal: IntArray get() = keys

    @PublishedApi internal val valuesInternal: DoubleArray get() = values

    @PublishedApi internal val usedInternal: BooleanArray get() = used

    private fun deleteSlot(start: Int) {
        var i = start
        used[i] = false
        var j = i
        while (true) {
            j = (j + 1) and mask
            if (!used[j]) return
            val home = mixIntKey(keys[j]) and mask
            if (mustStayDuringShift(home, i, j)) continue
            keys[i] = keys[j]
            values[i] = values[j]
            used[i] = true
            i = j
            used[i] = false
        }
    }

    private fun grow() {
        val oldKeys = keys
        val oldValues = values
        val oldUsed = used
        val newCap = keys.size * 2
        keys = IntArray(newCap)
        values = DoubleArray(newCap)
        used = BooleanArray(newCap)
        mask = newCap - 1
        for (i in oldUsed.indices) {
            if (!oldUsed[i]) continue
            var j = mixIntKey(oldKeys[i]) and mask
            while (used[j]) j = (j + 1) and mask
            used[j] = true
            keys[j] = oldKeys[i]
            values[j] = oldValues[i]
        }
    }
}

/** Next power-of-two capacity that keeps the load factor ≤ 0.5 for [initialCapacity] entries. */
internal fun openAddressingCapacity(initialCapacity: Int): Int {
    var cap = 8
    while (cap < initialCapacity * 2) cap *= 2
    return cap
}

/** Fibonacci-multiplicative hash; good distribution for sequential int keys. */
internal fun mixIntKey(x: Int): Int = x * -0x61c88647

/**
 * Backward-shift deletion predicate (Knuth 6.4 algorithm R): with a hole at `i` and the next
 * occupied slot at `j`, an entry whose home slot lies cyclically within `(i, j]` is still
 * reachable past the hole and must stay; otherwise it would be stranded and must shift down.
 */
internal fun mustStayDuringShift(home: Int, i: Int, j: Int): Boolean =
    if (i <= j) home > i && home <= j else home > i || home <= j
