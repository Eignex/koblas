package com.eignex.koblas.dense

/** Performance eligibility only; a rule never establishes capability or semantic eligibility. */
internal sealed interface WorkRule {
    fun accepts(rows: Int, columns: Int = 1, depth: Int = 1): Boolean

    data object Never : WorkRule {
        override fun accepts(rows: Int, columns: Int, depth: Int): Boolean = false
    }

    data object AlwaysEligible : WorkRule {
        override fun accepts(rows: Int, columns: Int, depth: Int): Boolean = true
    }

    data class Minimum(val work: Long) : WorkRule {
        init {
            require(work >= 0) { "negative minimum work" }
        }
        override fun accepts(rows: Int, columns: Int, depth: Int): Boolean = saturatedWork(rows, columns, depth) >= work
    }

    data class Shape(
        val minimumRows: Int,
        val minimumColumns: Int,
        val minimumDepth: Int,
        val maximumDepth: Int,
        val minimumWork: Long = 0,
    ) : WorkRule {
        init {
            require(minimumRows >= 0 && minimumColumns >= 0 && minimumDepth >= 0 && maximumDepth >= minimumDepth) {
                "invalid shape rule"
            }
            require(minimumWork >= 0) { "negative minimum work" }
        }
        override fun accepts(rows: Int, columns: Int, depth: Int): Boolean =
            rows >= minimumRows && columns >= minimumColumns && depth in minimumDepth..maximumDepth &&
                saturatedWork(rows, columns, depth) >= minimumWork
    }
}

/** Nonnegative operation work, saturated rather than wrapped when m * n * k exceeds Long capacity. */
internal fun saturatedWork(rows: Int, columns: Int = 1, depth: Int = 1): Long {
    require(rows >= 0 && columns >= 0 && depth >= 0) { "negative operation shape" }
    val area = rows.toLong() * columns
    return if (depth != 0 && area > Long.MAX_VALUE / depth) Long.MAX_VALUE else area * depth
}

/** Host paths have independently measured call costs. No ordering between the three rules is implied. */
internal data class HostCrossovers(val scalarToC: WorkRule, val simdToC: WorkRule, val nativeToC: WorkRule)
