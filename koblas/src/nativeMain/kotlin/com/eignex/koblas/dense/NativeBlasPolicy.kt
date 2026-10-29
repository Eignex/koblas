package com.eignex.koblas.dense

/** Native array pinning amortizes foreign calls without the JVM binding's operand transfers. */
internal object NativeBlasPolicy : DenseHostEligibility {
    // Conservative floor from the bounded Native GEMM/GEMV/TRSM ladder; see the benchmark guide.
    private const val MINIMUM_WORK = 1024L

    /** Measured Native Level 1 thresholds from the pinned oneMKL sweep on i9-12900H. */
    fun crossover(operation: DenseOperation): Int = when (operation) {
        DenseOperation.Nrm2, DenseOperation.Iamax -> 128
        else -> 64
    }

    override fun eligible(operation: DenseMatrixOperation, rows: Int, columns: Int, depth: Int): Boolean {
        if (rows <= 0 || columns <= 0 || depth <= 0 || operation == DenseMatrixOperation.GerWindow) return false
        val divisor = when (operation) {
            DenseMatrixOperation.Syr, DenseMatrixOperation.Trmv, DenseMatrixOperation.TrmvTransposed,
            DenseMatrixOperation.Trsv, DenseMatrixOperation.TrsvTransposed,
            DenseMatrixOperation.Gemmt, DenseMatrixOperation.Syrk,
            DenseMatrixOperation.Trmm, DenseMatrixOperation.Trsm,
            -> 2

            else -> 1
        }
        // Compare before multiplying by depth so diagnostic queries with huge extents cannot overflow.
        val minimumArea = (MINIMUM_WORK * divisor + depth - 1) / depth
        return rows.toLong() * columns >= minimumArea
    }
}
