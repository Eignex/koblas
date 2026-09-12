package com.eignex.koblas.dense

/** Scheduling dimensions are independent of legal native variants and compute microtiles. */
internal data class BlockSchedule(
    val rows: Int = 128,
    val columns: Int = 256,
    val depth: Int = 256,
    val leftGroup: Int = 4,
    val rightGroup: Int = 4,
    val diagonalBlock: Int = 64,
    val rhsBlock: Int = 32,
    val nativeCallWorkLimit: Long = 1_048_576,
) {
    init {
        require(rows > 0 && columns > 0 && depth > 0 && leftGroup > 0 && rightGroup > 0) { "invalid block geometry" }
        require(diagonalBlock in 1..Long.SIZE_BITS && rhsBlock > 0) { "invalid triangular schedule" }
        require(nativeCallWorkLimit > 0) { "native call work limit must be positive" }
        val left = PackedMatrixLayout(PackedRole.Left, rows, depth, leftGroup).storageSize
        val right = PackedMatrixLayout(PackedRole.Right, depth, columns, rightGroup).storageSize
        checkedStorageSize(left.toLong() + right)
    }
}
