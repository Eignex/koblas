package com.eignex.koblas

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
        installLinearAlgebra(ReferenceLinearAlgebra)
    }

    @AfterTest
    fun restoreSelection() {
        installLinearAlgebra(null)
    }

    /**
     * Best of several measurement windows. A single window is fragile: one unrelated event inside it — a
     * class load, a JIT recompilation — is divided by the iteration count and can exceed a per-iteration
     * budget on its own, which is how this first failed on CI while passing locally. An allocation-free
     * loop produces a zero window given enough tries; an allocating one cannot.
     */
    private fun bytesPerIteration(iterations: Int, warmup: Int = 200, block: (Int) -> Unit): Double {
        repeat(warmup) { block(it) } // let the JIT settle; first calls allocate profiling data
        val id = Thread.currentThread().threadId()
        var best = Double.MAX_VALUE
        repeat(WINDOWS) {
            val before = bean.getThreadAllocatedBytes(id)
            repeat(iterations) { i -> block(i) }
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
        assertTrue(into < 64.0, "solveInto allocated $into B per iteration (allocating form: $allocating B)")
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
        assertTrue(
            pooled < 64.0,
            "rcond with a workspace allocated $pooled B per iteration (plain: $allocating B)",
        )
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
        assertTrue(into < 64.0, "factorInto allocated $into B per iteration (allocating form: $allocating B)")
    }

    @Test
    fun `the symmetric rank-k mirror buffer is lent rather than allocated`() {
        val n = 64
        val rng = Random(20260741)
        val a = wellConditioned(n, rng)
        val c = DenseMatrix(n, n)
        // syrk's FULL mode needs an n² scratch, the largest single allocation in the library.
        val ws = Workspace().apply { reserve(n * n, count = 1) }

        val allocating = bytesPerIteration(200) { koblas.syrk(1.0, a, transpose = true, beta = 0.0, c = c) }
        val pooled = bytesPerIteration(200) {
            koblas.syrk(1.0, a, transpose = true, beta = 0.0, c = c, workspace = ws)
        }
        assertTrue(allocating > n * n * Double.SIZE_BYTES * 0.5, "expected an n² scratch, saw $allocating B")
        assertTrue(pooled < 64.0, "syrk allocated $pooled B per iteration (plain: $allocating B)")
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
        assertTrue(into < 64.0, "block solveInto allocated $into B per iteration (plain: $allocating B)")
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
        assertTrue(into < 64.0, "least squares allocated $into B per iteration (plain: $allocating B)")
    }

    @Test
    fun `norm1 with a workspace allocates nothing`() {
        val n = 64
        val a = wellConditioned(n, Random(20260742))
        val ws = Workspace().apply { reserve(n, count = 1) }
        val allocating = bytesPerIteration(2000) { norm1(a) }
        val pooled = bytesPerIteration(2000) { norm1(a, ws) }
        assertTrue(allocating > n * Double.SIZE_BYTES * 0.5, "expected allocation, saw $allocating B")
        assertTrue(pooled < 64.0, "norm1 allocated $pooled B per iteration (plain: $allocating B)")
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
        val basis = SparseLu.factorize(SparseMatrix.ofColumns(m, m, columns))!!
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val x = DoubleArray(m)
        val y = DoubleArray(m)
        val ws = Workspace()

        val allocating = bytesPerIteration(500) {
            basis.ftran(b)
            basis.btran(b)
        }
        val into = bytesPerIteration(500) {
            basis.ftranInto(b, x, ws)
            basis.btranInto(b, y, ws)
        }
        assertTrue(allocating > m * Double.SIZE_BYTES * 2.0, "expected allocation, saw $allocating B")
        assertTrue(into < 64.0, "sparse FTRAN+BTRAN allocated $into B per iteration (plain: $allocating B)")
    }
}
