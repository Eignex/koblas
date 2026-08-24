package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.solve
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue

class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        /** A pooled form must allocate at most this fraction of what the allocating form does. */
        const val POOLED_RATIO = 50.0
    }

    /** Pins the portable backend for the suite, since an FFM call allocates a MemorySegment wrapper per array. */
    @BeforeTest
    fun usePortableKernels() {
        installBackends(
            koblas.with(
                blas = F64ReferenceLinearAlgebra,
                lapack = F64ReferenceLinearAlgebra,
                sparseBlas = F64ReferenceSparseLinearAlgebra,
                sparseLu = F64ReferenceSparseLinearAlgebra,
                sparseKernels = F64ReferenceSparseLinearAlgebra,
            ),
        )
    }

    @AfterTest
    fun restoreSelection() {
        installBackends(null)
    }

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
        val anorm = a.norm1()
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
        val c = F64DenseMatrix(n, n)
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
        val b = F64DenseMatrix(n, nrhs)
        for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
        val out = F64DenseMatrix(n, nrhs)
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
        val a = F64DenseMatrix(m, n)
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

    @Test
    fun `norm1 allocates nothing`() {
        val n = 64
        val a = wellConditioned(n, Random(20260742))
        val bytes = bytesPerIteration(2000) { a.norm1() }
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
        val basis = F64SparseMatrix.ofColumns(m, m, columns).lu()
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

    /**
     * The symmetric rank-one and rank-two updates take a [F64VectorLike], and densifying a dense operand to
     * read it copies the whole vector on every call. They are the innermost step of a covariance or a
     * quasi-Newton update, so a copy per call is a copy per iteration of the caller's loop.
     */
    @Test
    fun `a symmetric rank update loop allocates nothing per iteration`() {
        val n = 96
        val rng = Random(20260824)
        val a = F64DenseMatrix.zero(n, n)
        val x = F64DenseVector.wrap(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
        val y = F64DenseVector.wrap(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
        val syr = bytesPerIteration(500) { koblas.syr(1e-12, x, a, Uplo.LOWER) }
        val syr2 = bytesPerIteration(500) { koblas.syr2(1e-12, x, y, a, Uplo.LOWER) }
        assertTrue(syr < FLOOR_BYTES, "syr allocated $syr B per call for a dense operand it can read in place")
        assertTrue(syr2 < FLOOR_BYTES, "syr2 allocated $syr2 B per call for dense operands it can read in place")
    }
}
