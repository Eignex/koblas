package com.eignex.koblas.dense

import com.eignex.koblas.internal.backend.tunedInt

/**
 * Every cache block size and dispatch crossover the dense routines apply, in one place and settable from
 * outside the build, as [com.eignex.koblas.sparse.SparseTuning] is for the sparse half.
 *
 * These are measured numbers rather than derived ones, and what they were measured on is one machine. A
 * deployment on a different cache hierarchy, or anyone retuning, needs to move them without editing the
 * source, so each resolves from a JVM system property, then an environment variable, then the compiled-in
 * default. The property is `koblas.dense.<name>` and the variable is its upper-case spelling with dots
 * replaced by underscores, matching the keys in `ConfigurationKeys`. There are no system properties off the
 * JVM, so a Native target reads the environment variable alone.
 *
 * Every value is read once, when this object initializes. Nothing here is consulted per call, which matters
 * because most of these sit inside a dispatch decision that runs on every vector.
 *
 * An override that is not an integer, or that falls outside the range its entry allows, is ignored and the
 * default stands. That follows how a configured library path that is not absolute is ignored rather than
 * resolved: a typo in a deployment's environment must not take a numerical library down at class load, and
 * a block size of zero or a negative crossover would do exactly that.
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

    /**
     * Columns of the product a portable level-3 block covers.
     *
     * Small because of the shape of the innermost loop rather than in spite of it. `Level3Benchmark.gemm`
     * at order 1024 measures 32, 64, 128 and 256 in turn at 1.3x, 1.3x, 1.3x and 2.5x the time this one
     * takes, so the temptation to widen it for reuse of the A panel has been tried and is a loss.
     */
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

    /**
     * Order at or above which a multi-column triangular solve blocks instead of walking its columns.
     *
     * From `BlockSolveBenchmark.blockSolve` on the reference backend at `nrhs = 32`, blocked against the
     * column loop: `n = 384` was a wash at 727 us against 709 with the error bars overlapping, `n = 512`
     * ran 1.22x faster at 1265 us against 1546, and `n = 1024` 1.29x faster at 5211 us against 6730, the
     * last two with error bars clear of one another. Below the crossover the triangle still fits in cache,
     * so re-reading it per column costs little and the blocking only adds its bookkeeping. This takes the
     * conservative end of the measured range.
     */
    val trsmBlockedMinOrder: Int = tuned("trsm.blocked.min.order", default = 512)

    /**
     * Smallest triangular order whose repeated packed updates repay packing both operands.
     *
     * End-to-end `TrsmBenchmark.denseTrsm` comparisons with packing forced on and off put order 15 ahead
     * for a 64-row panel, but retain 16 as the conservative tile-aligned boundary.
     */
    val trsmPackedMinOrder: Int = tuned("trsm.packed.min.order", default = 16)

    /**
     * Smallest normalized right-hand-side row count sent through packed TRSM tiles.
     *
     * Forced packed/scalar comparisons at order 32 lose below 16 rows, overlap through 31, and put packed
     * 1.12x to 1.59x ahead at 32 and 33 rows across the right-transposed and left-transposed orientations.
     */
    val trsmPackedMinRows: Int = tuned("trsm.packed.min.rows", default = 32)

    /**
     * Run length from which crossing into the bundled C library beats staying on the JVM.
     *
     * Every such call wraps each array in a MemorySegment and goes through invokeExact, which costs tens of
     * nanoseconds whatever the length, so a short run pays for a foreign call to do work the JIT would have
     * finished already. Measured on `Level1Benchmark` with these set to zero, so the C arm really crosses
     * rather than falling back to the same portable code the scalar arm runs. Across three runs the four
     * plain reductions are level or behind at 64 and ahead at 128. Past the crossover each pulls away,
     * reaching 2.8x to 4.1x by 2048.
     *
     * Only reductions have one. HotSpot will not vectorise a floating-point reduction, since splitting the
     * sum across lanes reorders the additions and changes the result, so those loops run an element at a
     * time however hot they get and the C kernel has something to beat. The elementwise routines never
     * cross at any length and so have nothing to tune here.
     */
    val jvmCDotCrossover: Int = tuned("jvm.c.dot.crossover", default = 128)

    /** Bundled C crossover for the plain sum, measured with [jvmCDotCrossover]. */
    val jvmCSumCrossover: Int = tuned("jvm.c.sum.crossover", default = 128)

    /**
     * Bundled C crossover for the sum of squared differences.
     *
     * Higher than the others because it reads two operands, which is the one measured case that only
     * separates at 256.
     */
    val jvmCSsqdCrossover: Int = tuned("jvm.c.ssqd.crossover", default = 256)

    /** Bundled C crossover for the euclidean norm, measured with [jvmCDotCrossover]. */
    val jvmCNrm2Crossover: Int = tuned("jvm.c.nrm2.crossover", default = 128)

    /** Bundled C crossover for the absolute sum, measured with [jvmCDotCrossover]. */
    val jvmCAsumCrossover: Int = tuned("jvm.c.asum.crossover", default = 128)

    /**
     * Bundled C crossover for the four-way dot.
     *
     * Four dots share one pass over the shared operand, so the same foreign call covers four runs of this
     * length. The point estimate leads from 256 upward and holds about 1.4x, but the JIT arm varies enough
     * that 512 is the shortest length where the two separate beyond their error bars, which is the same
     * criterion the other crossovers use. This is also the one kernel the AVX2 clones in
     * `koblas_kernels.h` leave alone, since the wider registers cost it 1.7x, so what it crosses into is
     * the baseline build.
     */
    val jvmCDot4Crossover: Int = tuned("jvm.c.dot4.crossover", default = 512)

    /**
     * Run length from which four vector accumulators beat one, counted in whole vectors so that the
     * threshold scales with the lane width. The multiplier is what is tunable; the lane count is a property
     * of the machine and is applied where this is read.
     *
     * Set where `jvmLevel1Benchmark` shows the win without argument rather than at the crossover: on four
     * lanes the unrolled form runs about 2x to 3x at 1024 and 4096, while unrolling every length is slower
     * than not unrolling below 32 elements. The scaling by lane width is reasoning, not measurement: only
     * four lanes were measured.
     */
    val simdUnrollMinVectors: Int = tuned("simd.unroll.min.vectors", default = 32)

    /**
     * Shortest run for which crossing into the C kernels pays on Kotlin/Native.
     *
     * Every such call pins its arrays and crosses a foreign boundary, and below the C library's own
     * unrolling threshold the kernel runs the same plain loop the scalar helper would, so a short run buys
     * the overhead and nothing else. Measured on `Level1Benchmark` for linuxX64, C against scalar: at len 2
     * dot is 40.7 ns against 19.2 and at 8 it is 39.8 against 22.4, both losses; at 32 they are level
     * (43.4 against 40.2); at 48 C is ahead on all three measured routines (dot 47.7 against 51.3, axpy
     * 39.2 against 49.2, nrm2 34.5 against 41.8) and pulls away from there, reaching 2x by 128. This takes
     * the conservative end of that range.
     */
    val nativeCMinLength: Int = tuned("native.c.min.length", default = 48)

    private fun tuned(name: String, default: Int, minimum: Int = 1, maximum: Int = Int.MAX_VALUE): Int =
        tunedInt(PREFIX, name, default, minimum, maximum)

    /** The segment every key in this collection carries after `koblas.`. */
    private const val PREFIX = "dense"
}
