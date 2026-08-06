package com.eignex.koblas.bench

import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.axpy
import com.eignex.koblas.dot
import com.eignex.koblas.koblas
import kotlin.random.Random
import kotlinx.benchmark.Benchmark
import kotlinx.benchmark.BenchmarkMode
import kotlinx.benchmark.BenchmarkTimeUnit
import kotlinx.benchmark.Mode
import kotlinx.benchmark.OutputTimeUnit
import kotlinx.benchmark.Param
import kotlinx.benchmark.Scope
import kotlinx.benchmark.Setup
import kotlinx.benchmark.State

/**
 * The sparse level-1 kernels across nonzero counts: `usdot` against a dense operand, against another sparse
 * one, `usaxpy`, and the scatter.
 *
 * The counterpart of [Level1Benchmark] for the sparse tier, and it exists for the same reason: these are the
 * routines a backend would replace one at a time, and no seam is worth installing without knowing what the
 * portable loop costs first. JVM-only because that is where the alternative to a portable loop is, so a
 * common suite would measure the same kernels twice on the native targets.
 *
 * Two axes, because an indexed kernel has one the dense kernels do not:
 *
 * - [nnz] is the loop trip count, the axis any vectorized replacement has to beat a short scalar loop on;
 * - [spread] is how far apart the touched positions are (`size = nnz · spread`). Reading four positions
 *   inside one cache line is a different operation in practice from reading four spread across four lines,
 *   and a real sparse column is the latter. `spread = 1` is the contiguous best case, kept as an upper bound
 *   rather than as a realistic shape.
 *
 * The resolved kernel name is printed because these routines are behind a seam: a host or platform sparse
 * kernel set would change what this measures without changing a line here.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class SparseLevel1Benchmark {
    @Param("8", "32", "128", "512", "2048")
    var nnz: Int = 0

    @Param("1", "16")
    var spread: Int = 0

    private lateinit var x: SparseVector
    private lateinit var other: SparseVector
    private lateinit var dense: DoubleArray
    private lateinit var denseView: DenseVector

    @Setup
    fun setup() {
        println("resolved: sparseVectorKernels=${koblas.sparseVectorKernels.name}")
        val rng = Random(BENCH_SEED)
        val size = nnz * spread
        x = sparseVector(size, nnz, rng)
        other = sparseVector(size, nnz, rng)
        dense = randomVector(size, rng)
        denseView = DenseVector.wrap(dense)
    }

    /** `usdot`: one gathered read of the dense operand per stored entry. */
    @Benchmark
    fun dotDense(): Double = x dot denseView

    /** Two sparse operands, whose index-list merge is a different cost from the gather above. */
    @Benchmark
    fun dotSparse(): Double = x dot other

    /** `usaxpy`: read, multiply-add, write back, all at the stored positions. */
    @Benchmark
    fun axpyDense() {
        axpy(denseView, 1.000001, x)
    }

    /** `ussc`: the pure write, with no arithmetic to hide its cost behind. */
    @Benchmark
    fun scatterDense() {
        koblas.sparseVectorKernels.scatter(x, dense)
    }
}

/**
 * A sparse vector of exactly [nnz] entries, one per `size / nnz` block with a random position inside it.
 *
 * One per block rather than [nnz] draws from the whole range, so the count is exact and the ascent the
 * `SparseVector` constructor requires holds by construction: entry `k` sits in `[k·stride, (k+1)·stride)`,
 * which cannot reach entry `k+1`'s block. Drawing independently would need a dedup, and deduping would make
 * the realized nnz — the axis being swept — depend on the seed.
 */
internal fun sparseVector(size: Int, nnz: Int, rng: Random): SparseVector {
    require(nnz <= size) { "nnz $nnz exceeds size $size" }
    val stride = size / nnz
    val indices = IntArray(nnz) { k -> k * stride + rng.nextInt(stride) }
    return SparseVector.of(size, indices, DoubleArray(nnz) { rng.nextDouble(-1.0, 1.0) })
}
