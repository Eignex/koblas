package com.eignex.koblas

import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.lu
import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The point of the destination-passing forms is that a steady-state loop stops allocating, so that is
 * measured rather than asserted. `ThreadMXBean` reports bytes allocated by this thread, which counts
 * every heap allocation the loop makes.
 *
 * The budget is per iteration and generous: the aim is to catch a routine that allocates *per call* — an
 * `n`-vector, an `n²` copy — not to police a few bytes of boxing. A simplex iteration at `n` 64 that
 * allocated one vector per solve would show ~1 KB per iteration and fail immediately.
 */
class AllocationFreeTest {

    private companion object {
        const val WINDOWS = 5

        /** Allowance for effects that are not koblas's: instrumentation, index boxing, JIT noise. */
        const val FLOOR_BYTES = 64.0

        /** A pooled form must allocate at most this fraction of what the allocating form does. */
        const val POOLED_RATIO = 50.0
    }

    private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    /**
     * Pins the portable backend for the whole suite.
     *
     * What is under test is koblas's own scratch handling — that a [Workspace] replaces the temporaries
     * these routines would otherwise allocate. A host BLAS has its own allocation profile (the FFM
     * binding wraps each array in a `MemorySegment` per call, and its `syrk` needs no `n²` scratch at
     * all), so leaving it installed would measure the wrong thing and, on a machine that happens to have
     * OpenBLAS, fail for the wrong reason.
     */
    @BeforeTest
    fun usePortableKernels() {
        // The sparse halves are pinned too, not just the dense ones. The allocation budget here is a promise
        // about koblas's OWN routines; a host backend cannot make it, since an FFM call allocates a
        // MemorySegment wrapper per array however carefully the rest is pooled. Before UMFPACK existed this
        // line only needed the dense halves, and the sparse ones silently stayed portable.
        installBackends(
            koblas.with(
                blas = ReferenceLinearAlgebra,
                lapack = ReferenceLinearAlgebra,
                sparseBlas = ReferenceSparseLinearAlgebra,
                sparseLapack = ReferenceSparseLinearAlgebra,
                sparseVectorKernels = ReferenceSparseLinearAlgebra,
            ),
        )
    }

    @AfterTest
    fun restoreSelection() {
        installBackends(null)
    }

    /**
     * Holds each block's result so escape analysis cannot delete the allocation being measured.
     *
     * Without this the baseline halves of these tests are unreliable: a result that is computed and
     * discarded is a candidate for scalar replacement, and a `solve` whose fresh matrix never escapes can
     * be optimized into no allocation at all. That is not hypothetical — it passed here and failed on CI
     * after the JVM target moved to 25, which is exactly the shape of a JIT-dependent flake.
     */
    private var sink: Any? = null

    /**
     * Best of several measurement windows. A single window is fragile: one unrelated event inside it — a
     * class load, a JIT recompilation — is divided by the iteration count and can exceed a per-iteration
     * budget on its own, which is how this first failed on CI while passing locally. An allocation-free
     * loop produces a zero window given enough tries; an allocating one cannot.
     */
    private fun bytesPerIteration(iterations: Int, warmup: Int = 200, block: (Int) -> Any?): Double {
        repeat(warmup) { sink = block(it) } // let the JIT settle; first calls allocate profiling data
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(WINDOWS) {
            val before = bean.getThreadAllocatedBytes(id)
            repeat(iterations) { i -> sink = block(i) }
            val after = bean.getThreadAllocatedBytes(id)
            val perIteration = (after - before).toDouble() / iterations
            if (perIteration < best) best = perIteration
        }
        return best
    }

    private fun wellConditioned(n: Int, rng: Random): DenseMatrix {
        val a = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
        }
        return a
    }

    /**
     * Asserts that the destination-passing or pooled form removed the temporary the allocating form pays.
     *
     * Bounded as a fraction of the allocating form rather than by an absolute byte count. The temporaries
     * at stake are `O(n)` to `O(n·nrhs)` arrays — kilobytes — so a regression that stops pooling shows up
     * as a ratio near 1, while an absolute budget of a few dozen bytes is sensitive to things that have
     * nothing to do with koblas: coverage instrumentation, boxing of a loop index, and how aggressively a
     * particular JIT eliminates short-lived objects. That sensitivity is not theoretical — a 64-byte
     * budget passed here and failed on CI.
     */
    private fun assertPooled(pooled: Double, allocating: Double, what: String) {
        val budget = maxOf(FLOOR_BYTES, allocating / POOLED_RATIO)
        assertTrue(
            pooled < budget,
            "$what allocated $pooled B per iteration against a $budget B budget (allocating form: $allocating B)",
        )
    }

    @Test
    fun `a dense solve loop allocates nothing per iteration`() {
        val n = 64
        val rng = Random(20260736)
        val lu = wellConditioned(n, rng).lu()
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(n)

        val allocating = bytesPerIteration(2000) { koblas.solve(lu, b) }
        val into = bytesPerIteration(2000) { koblas.solveInto(lu, b, x) }
        assertTrue(
            allocating > n * Double.SIZE_BYTES * 0.5,
            "expected the allocating form to allocate, saw $allocating B",
        )
        assertPooled(into, allocating, "solveInto")
    }

    @Test
    fun `a condition estimate with a workspace allocates nothing per iteration`() {
        val n = 48
        val rng = Random(20260737)
        val a = wellConditioned(n, rng)
        val anorm = norm1(a)
        val lu = a.lu()
        val ws = Workspace()

        val allocating = bytesPerIteration(500) { koblas.rcond(lu, anorm) }
        val pooled = bytesPerIteration(500) { koblas.rcond(lu, anorm, ws) }
        assertTrue(allocating > n * Double.SIZE_BYTES * 2.0, "expected allocation, saw $allocating B")
        assertPooled(pooled, allocating, "rcond")
    }

    @Test
    fun `refactorizing in place allocates nothing per iteration`() {
        val n = 48
        val rng = Random(20260738)
        val a = wellConditioned(n, rng)
        val reused = koblas.factor(a)

        val allocating = bytesPerIteration(300) { koblas.factor(a) }
        val into = bytesPerIteration(300) { koblas.factorInto(a, reused) }
        assertTrue(allocating > n * n * Double.SIZE_BYTES * 0.5, "expected an n² copy, saw $allocating B")
        assertPooled(into, allocating, "factorInto")
    }

    @Test
    fun `the symmetric rank-k mirror buffer is lent rather than allocated`() {
        val n = 64
        val rng = Random(20260741)
        val a = wellConditioned(n, rng)
        val c = DenseMatrix(n, n)
        // syrk's FULL mode needs an n² scratch, the largest single allocation in the library. It is the
        // rank-1-sweep branch that needs it, which under column-major storage is the non-transposed
        // direction: C = A·Aᵀ sweeps columns of A, where C = Aᵀ·A is a plain dot per entry.
        val ws = Workspace().apply { reserve(n * n, count = 1) }

        val allocating = bytesPerIteration(200) { koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c) }
        val pooled = bytesPerIteration(200) {
            koblas.syrk(1.0, a, transpose = false, beta = 0.0, c = c, workspace = ws)
        }
        assertTrue(allocating > n * n * Double.SIZE_BYTES * 0.5, "expected an n² scratch, saw $allocating B")
        assertPooled(pooled, allocating, "syrk")
    }

    @Test
    fun `a block solve loop allocates nothing per iteration`() {
        val n = 48
        val nrhs = 40 // above the crossover, so the blocked path runs rather than column-by-column
        val rng = Random(20260743)
        val lu = wellConditioned(n, rng).lu()
        val b = DenseMatrix(n, nrhs)
        for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
        val out = DenseMatrix(n, nrhs)
        val ws = Workspace().apply { reserve(n * nrhs, count = 1) }

        val allocating = bytesPerIteration(200) { koblas.solve(lu, b) }
        val into = bytesPerIteration(200) { koblas.solveInto(lu, b, out, workspace = ws) }
        assertTrue(allocating > n * nrhs * Double.SIZE_BYTES * 0.5, "expected allocation, saw $allocating B")
        assertPooled(into, allocating, "block solveInto")
    }

    @Test
    fun `a least-squares loop allocates nothing per iteration`() {
        val m = 64
        val n = 16
        val rng = Random(20260744)
        val a = DenseMatrix(m, n)
        for (i in 0 until m) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
        val f = koblas.qr(a)
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(n)
        val ws = Workspace().apply { reserve(m, count = 1) }

        val allocating = bytesPerIteration(500) { koblas.solveLeastSquares(f, b) }
        val into = bytesPerIteration(500) { koblas.solveLeastSquaresInto(f, b, x, ws) }
        assertTrue(allocating > m * Double.SIZE_BYTES * 0.5, "expected allocation, saw $allocating B")
        assertPooled(into, allocating, "least squares")
    }

    /**
     * norm1 allocates nothing at all, and takes no workspace to manage.
     *
     * Under row-major storage it accumulated one running total per column while sweeping rows, so it
     * needed an `n`-wide array and a workspace was the way to avoid allocating one. A column is
     * contiguous here, so each column sum completes before the next begins and a single scalar suffices.
     * The parameter is gone; this asserts the stronger property that replaced it.
     */
    @Test
    fun `norm1 allocates nothing`() {
        val n = 64
        val a = wellConditioned(n, Random(20260742))
        val bytes = bytesPerIteration(2000) { norm1(a) }
        assertTrue(bytes <= FLOOR_BYTES, "norm1 allocated $bytes B")
    }

    @Test
    fun `a sparse simplex-shaped iteration allocates nothing`() {
        val m = 64
        val rng = Random(20260739)
        val columns = List(m) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-1.0, 1.0) + m))
            for (i in 0 until m) if (i != j && rng.nextDouble() < 0.05) entries.add(i to rng.nextDouble(-1.0, 1.0))
            entries
        }
        val basis = SparseMatrix.ofColumns(m, m, columns).lu()
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(m)
        val y = DoubleArray(m)
        val ws = Workspace()

        val allocating = bytesPerIteration(500) {
            basis.solve(b)
            basis.solve(b, transpose = true)
        }
        val into = bytesPerIteration(500) {
            basis.solveInto(b, x, workspace = ws)
            basis.solveInto(b, y, transpose = true, workspace = ws)
        }
        assertTrue(allocating > m * Double.SIZE_BYTES * 2.0, "expected allocation, saw $allocating B")
        assertPooled(into, allocating, "sparse solve both directions")
    }
}
