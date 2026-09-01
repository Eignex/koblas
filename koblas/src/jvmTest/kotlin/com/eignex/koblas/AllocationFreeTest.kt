package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.REFERENCE_SPARSE_RHS_WIDTH
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.sparse.sparseConformanceSystem
import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.random.Random
import kotlin.test.*

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
                kernels = F64ScalarKernels,
                blas = F64ReferenceLinearAlgebra,
                decompositions = F64ReferenceLinearAlgebra,
                sparseBlas = F64ReferenceSparseLinearAlgebra,
                sparseDecompositions = F64ReferenceSparseLinearAlgebra,
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
        val into = bytesPerIteration(2000) { lu.solveInto(b, x) }
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
        val a = F64DenseMatrix(2 * n, n)
        for (j in 0 until n) {
            for (i in 0 until 2 * n) {
                a[i, j] = if (i == j) 2.0 else rng.nextDouble(-1.0, 1.0)
            }
        }
        val reused = koblas.factor(a)

        val allocating = bytesPerIteration(300) { koblas.factor(a) }
        val into = bytesPerIteration(300) { reused.refactorInto(a) }
        assertTrue(allocating > n * n * Double.SIZE_BYTES, "expected a rectangular factor copy, saw $allocating B")
        assertPooled(into, allocating, "factorInto")
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
        val into = bytesPerIteration(200) { lu.solveInto(b, out, workspace = ws) }
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

        val allocating = bytesPerIteration(500) { koblas.solve(f, b) }
        val into = bytesPerIteration(500) { f.solveInto(b, x, workspace = ws) }
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

    @Test
    fun `strict sparse allocation checks are allocation neutral`() {
        val n = 64
        val factor = sparseConformanceSystem(n, Random(20260828)).lu()
        val b = DoubleArray(n) { it * 0.01 - 0.5 }
        val out = DoubleArray(n)
        val workspace = Workspace().apply { reserve(n, count = 2) }

        val bytes = bytesPerIteration(500) {
            factor.solveInto(
                b,
                out,
                workspace = workspace,
                allocationPolicy = AllocationPolicy.REQUIRE_NO_MANAGED_OR_NATIVE,
            )
        }

        assertTrue(bytes <= FLOOR_BYTES, "strict sparse solve allocated $bytes B per call")
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
        val syr = bytesPerIteration(500) { koblas.syr(1e-12, x, a, lower = true) }
        val syr2 = bytesPerIteration(500) { koblas.syr2(1e-12, x, y, a, lower = true) }
        assertTrue(syr < FLOOR_BYTES, "syr allocated $syr B per call for a dense operand it can read in place")
        assertTrue(syr2 < FLOOR_BYTES, "syr2 allocated $syr2 B per call for dense operands it can read in place")
    }

    @Test
    fun `a transposed dense gemv workspace is allocation neutral`() {
        val rows = 64
        val columns = 72
        val a = F64DenseMatrix(rows, columns)
        val x = DoubleArray(rows) { it * 0.01 }
        val y = DoubleArray(columns)
        val workspace = Workspace().apply { reserve(4, count = 1) }

        val bytes = bytesPerIteration(1_000) {
            koblas.gemv(1e-8, a, x, 1.0, y, transpose = true, workspace = workspace)
            y
        }

        assertTrue(bytes <= FLOOR_BYTES, "transposed gemv allocated $bytes B per call")
    }

    @Test
    fun `transposed dense gemm workspace is allocation neutral`() {
        val m = 32
        val k = 48
        val n = 40
        val a = F64DenseMatrix(k, m)
        val b = F64DenseMatrix(n, k)
        val c = F64DenseMatrix(m, n)
        val workspace = Workspace().apply {
            reserve(k * n, count = 1)
            reserve(4, count = 1)
        }

        val bytes = bytesPerIteration(300) {
            koblas.gemm(1e-8, a, true, b, true, 1.0, c, workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "transposed gemm allocated $bytes B per call")
    }

    @Test
    fun `left symmetric matrix workspace is allocation neutral`() {
        val n = 96
        val a = F64DenseMatrix.diagonal(n)
        val b = F64DenseMatrix(n, 12)
        val c = F64DenseMatrix(n, 12)
        val workspace = Workspace().apply { reserve(n * n, count = 1) }

        val bytes = bytesPerIteration(300) {
            koblas.symm(1e-8, a, b, 1.0, c, workspace = workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "left symm allocated $bytes B per call")
    }

    @Test
    fun `right dense triangular solve workspace is allocation neutral`() {
        val n = 80
        val triangle = F64DenseMatrix.diagonal(n)
        val b = F64DenseMatrix(20, n)
        val workspace = Workspace().apply { reserve(n, count = 1) }

        val bytes = bytesPerIteration(500) {
            triangle.trsm(b, lower = true, right = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "right dense trsm allocated $bytes B per call")
    }

    @Test
    fun `sparse dense product workspace is allocation neutral`() {
        val n = 64
        val rows = 16
        val sparse = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = F64DenseMatrix(n, rows)
        val c = F64DenseMatrix(rows, n)
        val workspace = Workspace().apply { reserve(b.data.size, count = 1) }

        val bytes = bytesPerIteration(300) {
            koblas.sparseBlas.gemm(1e-8, sparse, false, b, true, 1.0, c, right = true, workspace = workspace)
            c
        }

        assertTrue(bytes <= FLOOR_BYTES, "right transposed sparse gemm allocated $bytes B per call")
    }

    @Test
    fun `left sparse triangular solve workspace is allocation neutral`() {
        val n = 64
        val sparse = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
        val b = F64DenseMatrix(n, 12)
        val workspace = Workspace().apply {
            reserve(n, count = 1)
            reserve(REFERENCE_SPARSE_RHS_WIDTH, count = 1)
        }

        val bytes = bytesPerIteration(500) {
            koblas.sparseBlas.trsm(sparse, b, lower = true, workspace = workspace)
            b
        }

        assertTrue(bytes <= FLOOR_BYTES, "left sparse trsm allocated $bytes B per call")
    }
}
