package com.eignex.koblas

import com.sun.management.ThreadMXBean
import java.lang.management.ManagementFactory
import kotlin.random.Random
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

    private val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean

    private fun bytesPerIteration(iterations: Int, warmup: Int = 200, block: (Int) -> Unit): Double {
        repeat(warmup) { block(it) } // let the JIT settle; first calls allocate profiling data
        val id = Thread.currentThread().threadId()
        val before = bean.getThreadAllocatedBytes(id)
        repeat(iterations) { block(it) }
        val after = bean.getThreadAllocatedBytes(id)
        return (after - before).toDouble() / iterations
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
    fun `a filter-shaped Cholesky update loop allocates nothing`() {
        val n = 48
        val rng = Random(20260740)
        val spd = wellConditioned(n, rng).let { a ->
            val m = DenseMatrix(n, n)
            koblas.syrk(1.0, a, transpose = true, beta = 0.0, c = m)
            for (i in 0 until n) m[i, i] = m[i, i] + n
            m
        }
        val l = spd.cholesky()
        val x = DenseVector.of(DoubleArray(n) { rng.nextDouble(-0.01, 0.01) })
        val ws = Workspace().apply { reserve(n, count = 3) }

        val allocating = bytesPerIteration(500) { l.choleskyUpdateInPlace(x) }
        val pooled = bytesPerIteration(500) { l.choleskyUpdateInPlace(x, ws) }
        assertTrue(allocating > n * Double.SIZE_BYTES * 2.0, "expected allocation, saw $allocating B")
        assertTrue(pooled < 64.0, "cholesky update allocated $pooled B per iteration (plain: $allocating B)")
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
