package com.eignex.koblas.sparse.internal

internal class MutableIntSet(initialCapacity: Int = 8) {
    /** Entry index plus one, so 0 reads as empty. */
    private var table: IntArray

    private var entries: IntArray
    private var mask: Int

    /** Number of members, and the exclusive bound on a valid slot. */
    var size: Int = 0
        private set

    init {
        val cap = openAddressingCapacity(initialCapacity)
        table = IntArray(cap)
        entries = IntArray(entryCapacityFor(cap))
        mask = cap - 1
    }

    /** How many members fit before the next growth, the width [MutableIntDoubleMap] matches. */
    val entryCapacity: Int get() = entries.size

    /** Slot of [key] among the entries, or `-1` if absent. Slots are invalidated by any mutation. */
    fun slotOf(key: Int): Int {
        var i = mixIntKey(key) and mask
        while (true) {
            val e = table[i]
            if (e == 0) return -1
            if (entries[e - 1] == key) return e - 1
            i = (i + 1) and mask
        }
    }

    fun keyAt(slot: Int): Int = entries[slot]

    /** Adds [key] if absent; returns its slot either way. */
    fun add(key: Int): Int {
        var i = mixIntKey(key) and mask
        while (true) {
            val e = table[i]
            if (e == 0) break
            if (entries[e - 1] == key) return e - 1
            i = (i + 1) and mask
        }
        val slot = size
        entries[slot] = key
        table[i] = slot + 1
        size++
        if (size * 2 > table.size) grow()
        return slot
    }

    /** Removes [key]; true if it was present. */
    fun remove(key: Int): Boolean {
        val slot = slotOf(key)
        if (slot < 0) return false
        removeAt(slot)
        return true
    }

    /**
     * Removes the member at [slot], filling the gap with the last one.
     *
     * @return the slot vacated by that move, or `-1` when the removed member was itself last.
     */
    fun removeAt(slot: Int): Int {
        var i = mixIntKey(entries[slot]) and mask
        while (table[i] != slot + 1) i = (i + 1) and mask
        deleteTableSlot(i)
        val last = size - 1
        size = last
        if (slot == last) return -1
        val moved = entries[last]
        entries[slot] = moved
        var j = mixIntKey(moved) and mask
        while (table[j] != last + 1) j = (j + 1) and mask
        table[j] = slot + 1
        return last
    }

    /** Invoke [action] for each member. The set must not be structurally modified during iteration. */
    inline fun forEach(action: (key: Int) -> Unit) {
        val e = entriesInternal
        for (s in 0 until size) action(e[s])
    }

    @PublishedApi internal val entriesInternal: IntArray get() = entries

    private fun deleteTableSlot(start: Int) {
        var i = start
        table[i] = 0
        var j = i
        while (true) {
            j = (j + 1) and mask
            val e = table[j]
            if (e == 0) return
            val home = mixIntKey(entries[e - 1]) and mask
            if (mustStayDuringShift(home, i, j)) continue
            table[i] = e
            table[j] = 0
            i = j
        }
    }

    private fun grow() {
        val newCap = table.size * 2
        table = IntArray(newCap)
        mask = newCap - 1
        entries = entries.copyOf(entryCapacityFor(newCap))
        for (s in 0 until size) {
            var i = mixIntKey(entries[s]) and mask
            while (table[i] != 0) i = (i + 1) and mask
            table[i] = s + 1
        }
    }
}

internal class MutableIntDoubleMap(initialCapacity: Int = 8) {
    @PublishedApi internal val keys: MutableIntSet = MutableIntSet(initialCapacity)

    private var values = DoubleArray(keys.entryCapacity)

    val size: Int get() = keys.size

    fun getOrDefault(key: Int, default: Double): Double {
        val slot = keys.slotOf(key)
        return if (slot >= 0) values[slot] else default
    }

    /** Slot of [key], or `-1` if absent. Slots are invalidated by any mutation. */
    fun slotOf(key: Int): Int = keys.slotOf(key)

    /** Value at [slot], which must come from a matching [slotOf]. */
    fun valueAt(slot: Int): Double = values[slot]

    fun setValueAt(slot: Int, value: Double) {
        values[slot] = value
    }

    fun put(key: Int, value: Double) {
        val slot = keys.add(key)
        if (values.size < keys.entryCapacity) values = values.copyOf(keys.entryCapacity)
        values[slot] = value
    }

    /** Remove [key]; returns true if it was present. */
    fun remove(key: Int): Boolean {
        val slot = keys.slotOf(key)
        if (slot < 0) return false
        removeAt(slot)
        return true
    }

    /** Remove the entry at [slot], which must come from a matching [slotOf]. */
    fun removeAt(slot: Int) {
        val moved = keys.removeAt(slot)
        if (moved >= 0) values[slot] = values[moved]
    }

    fun scaleValues(factor: Double) {
        for (slot in 0 until keys.size) values[slot] *= factor
    }

    /** Invoke [action] for each entry. The map must not be structurally modified during iteration. */
    inline fun forEach(action: (key: Int, value: Double) -> Unit) {
        val k = keys.entriesInternal
        val v = valuesInternal
        for (slot in 0 until keys.size) action(k[slot], v[slot])
    }

    @PublishedApi internal val valuesInternal: DoubleArray get() = values
}

/** Next power-of-two capacity that keeps the load factor ≤ 0.5 for [initialCapacity] entries. */
internal fun openAddressingCapacity(initialCapacity: Int): Int {
    var cap = 8
    while (cap < initialCapacity * 2) cap *= 2
    return cap
}

/** Members a table of [tableCapacity] holds before growth, which triggers just past half full. */
private fun entryCapacityFor(tableCapacity: Int): Int = tableCapacity / 2 + 1

/** Fibonacci-multiplicative hash, which distributes sequential int keys well. */
internal fun mixIntKey(x: Int): Int = x * -0x61c88647

/**
 * Backward-shift deletion predicate. With a hole at [i] and the next occupied slot at [j], an entry whose
 * home slot lies cyclically within `(i, j]` stays; anything else would be stranded and shifts down.
 */
internal fun mustStayDuringShift(home: Int, i: Int, j: Int): Boolean =
    if (i <= j) home > i && home <= j else home > i || home <= j
