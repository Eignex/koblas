package com.eignex.koblas.dense

import com.eignex.koblas.internal.configuration.tunedInt

/**
 * Every cache block size and dispatch crossover the dense routines apply, in one place and settable from
 * outside the build, as [com.eignex.koblas.sparse.SparseTuning] is for the sparse half.
 *
 * Each value resolves once from a JVM property, then an environment variable, then its default. Invalid
 * overrides are ignored so a deployment typo cannot prevent initialization.
 */
internal object DenseTuning {
    /**
     * Rows of the product a portable level-3 block covers.
     *
     * One A column is read across the columns of C the block covers, so the live piece of C is
     * [level3BlockRows] by [level3BlockColumns] doubles, 16 KB at these values, which is what fits beside
     * the A column in a first-level cache.
     */
    val level3BlockRows: Int = tuned("level3.block.rows", default = 256)

    /** Columns of the product a portable level-3 block covers. */
    val level3BlockColumns: Int = tuned("level3.block.columns", default = 8)

    /** Steps of the shared dimension a portable level-3 block accumulates before moving on. */
    val level3BlockDepth: Int = tuned("level3.block.depth", default = 128)

    /**
     * Diagonal block width for the blocked triangular routines.
     *
     * Bounded above by [Long.SIZE_BITS] rather than open: the trsm zero-pivot guard carries one bit per row
     * of a block in a `Long` mask, and Kotlin's `shl` reads only the low six bits of its operand, so a
     * wider block would wrap row `j` onto bit `j % 64` and mask the wrong rows. An override above that
     * bound is ignored. [requireTriangularBlockFitsMask] holds the same link for the default itself, which
     * this cannot check because a default is what an out-of-range override falls back to.
     */
    val triangularBlock: Int = tuned("triangular.block", default = 64, maximum = Long.SIZE_BITS)

    /**
     * Steps of the shared dimension the packed matrix product accumulates before it touches C again.
     *
     * This sets the working set, since both packed panels are sized from it: a panel of this depth by
     * [packedBlockColumns] and one of [packedBlockRows] by this depth are live at once. The tile of C
     * stays in registers for the whole of it, so a larger value means fewer passes over C and a larger
     * pair of panels to keep resident.
     */
    val packedBlockDepth: Int = tuned("packed.block.depth", default = 256)

    /** Rows of the left operand the packed matrix product copies into panels at a time. */
    val packedBlockRows: Int = tuned("packed.block.rows", default = 128)

    /** Columns of the right operand the packed matrix product copies into panels at a time. */
    val packedBlockColumns: Int = tuned("packed.block.columns", default = 256)

    /** Side of the square tile the blocked transpose moves at a time. */
    val transposeBlock: Int = tuned("transpose.block", default = 32)

    /** Smallest triangular order sent through packed updates. */
    val trsmPackedMinOrder: Int = tuned("trsm.packed.min.order", default = 16)

    /** Smallest right-hand-side row count sent through packed TRSM tiles. */
    val trsmPackedMinRows: Int = tuned("trsm.packed.min.rows", default = 32)

    /** Smallest triangular order sent through the packed matrix-product scheduler. */
    val trmmPackedMinOrder: Int = tuned("trmm.packed.min.order", default = 16)

    /** Smallest panel width sent through packed TRMM tiles. */
    val trmmPackedMinRows: Int = tuned("trmm.packed.min.rows", default = 32)

    /** Run length from which crossing into the bundled C library is selected. */
    val jvmCDotCrossover: Int = tuned("jvm.c.dot.crossover", default = 128)

    /** Bundled C crossover for the plain sum. */
    val jvmCSumCrossover: Int = tuned("jvm.c.sum.crossover", default = 128)

    /** Bundled C crossover for the sum of squared differences. */
    val jvmCSsqdCrossover: Int = tuned("jvm.c.ssqd.crossover", default = 256)

    /** Bundled C crossover for the euclidean norm. */
    val jvmCNrm2Crossover: Int = tuned("jvm.c.nrm2.crossover", default = 128)

    /** Bundled C crossover for the first maximum magnitude index. */
    val jvmCIamaxCrossover: Int = tuned("jvm.c.iamax.crossover", default = 512)

    /** JVM SIMD crossover for the first maximum magnitude index. */
    val simdIamaxCrossover: Int = tuned("simd.iamax.crossover", default = 256)

    /** Bundled C crossover for the absolute sum. */
    val jvmCAsumCrossover: Int = tuned("jvm.c.asum.crossover", default = 128)

    /** Bundled C crossover for the four-way dot. */
    val jvmCDot4Crossover: Int = tuned("jvm.c.dot4.crossover", default = 512)

    /** Bundled C crossover for the four-column AXPY used by non-transposed GEMV. */
    val jvmCAxpy4Crossover: Int = tuned("jvm.c.axpy4.crossover", default = 64)

    /** Bundled C crossover for the fused dot and AXPY used by SYMV. */
    val jvmCDotAxpyCrossover: Int = tuned("jvm.c.dot.axpy.crossover", default = 256)

    /**
     * Smallest symmetric order that shares four adjacent columns across [DensePanelKernels.dot4] and
     * [DensePanelKernels.axpy4].
     * Below this point the extra pass over the stored triangle costs more than the saved vector traffic.
     */
    val symvFourColumnCrossover: Int = tuned("symv.four.column.crossover", default = 512)

    /** Shared depth from which the bundled C four-by-four product tile is selected on the JVM. */
    val jvmCGemmTileCrossover: Int = tuned("jvm.c.gemm.tile.crossover", default = 16)

    /** Shared depth from which the bundled C fused packed update and solve is selected on the JVM. */
    val jvmCGemmTrsmTileCrossover: Int = tuned("jvm.c.gemm.trsm.tile.crossover", default = 16)

    /** Vector count from which four SIMD accumulators are selected. */
    val simdUnrollMinVectors: Int = tuned("simd.unroll.min.vectors", default = 32)

    /** Shortest Kotlin/Native run sent to the C kernels. */
    val nativeCMinLength: Int = tuned("native.c.min.length", default = 48)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "dense"
}
