package com.eignex.koblas.internal.engine

import com.eignex.koblas.engine.EngineImplementation

/** An internal crossover: minimum extents and work, optionally restricted by layout. */
internal data class EngineRule(
    val source: String,
    val implementations: Set<EngineImplementation>,
    val rows: Int = 0,
    val columns: Int = 0,
    val depth: Int = 0,
    val work: Long = 0,
    val contiguous: Boolean? = null,
    val borrowed: Boolean? = null,
) {
    val unconditional: Boolean get() = rows == 0 && columns == 0 && depth == 0 &&
        work == 0L && contiguous == null && borrowed == null

    fun matches(r: Int, c: Int, d: Int, adjacent: Boolean, view: Boolean): Boolean {
        if (r < rows || c < columns || d < depth ||
            (contiguous != null && contiguous != adjacent) || (borrowed != null && borrowed != view)
        ) {
            return false
        }
        if (work == 0L) return true
        if (r <= 0 || c <= 0 || d <= 0) return false
        // Divide the work threshold before comparing so even three maximum extents cannot overflow.
        return r.toLong() * c >= work / d + if (work % d == 0L) 0 else 1
    }
}
