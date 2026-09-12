package com.eignex.koblas.dense

/** Physical operand modes share the same block arithmetic; retained operands need no fresh packing. */
internal enum class ProductPacking { Direct, Left, Right, Both }

/** Algorithm choice is separate from native eligibility and host-call crossovers. */
internal data class ProductPolicy(
    val packLeft: WorkRule = WorkRule.Never,
    val packRight: WorkRule = WorkRule.Never,
    val packBoth: WorkRule = WorkRule.Never,
) {
    fun packing(rows: Int, columns: Int, depth: Int, retainedLeft: Boolean, retainedRight: Boolean): ProductPacking {
        require(rows >= 0 && columns >= 0 && depth >= 0) { "negative product shape" }
        if (rows == 0 || columns == 0 || depth == 0) return ProductPacking.Direct
        val both = packBoth.accepts(rows, columns, depth)
        val left = retainedLeft || both || packLeft.accepts(rows, columns, depth)
        val right = retainedRight || both || packRight.accepts(rows, columns, depth)
        return when {
            left && right -> ProductPacking.Both
            left -> ProductPacking.Left
            right -> ProductPacking.Right
            else -> ProductPacking.Direct
        }
    }
}
