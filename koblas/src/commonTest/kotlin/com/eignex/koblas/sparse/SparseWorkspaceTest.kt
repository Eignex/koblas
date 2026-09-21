package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * That a workspace handed to a sparse Level 2 routine is used. The answers these calls produce are
 * [SparseAliasTest]'s subject; what is checked here is that the snapshot an overlap forces comes from the
 * loan rather than from a fresh array, which is invisible to a caller reading the result.
 */
class SparseWorkspaceTest {

    /**
     * A square matrix storing one entry per column, so its coefficient array is exactly as long as the
     * vectors and either of them can be the buffer a call writes.
     */
    private fun permutation(): SparseMatrix = SparseMatrix.ofColumns(
        4,
        4,
        listOf(listOf(1 to 2.0), listOf(3 to -3.0), listOf(0 to 5.0), listOf(2 to 7.0)),
    )

    /**
     * A lower triangle storing as many coefficients as it has rows, which is what lets its values be the
     * right-hand side a triangular routine writes.
     */
    private fun triangle(): SparseMatrix =
        SparseMatrix.wrap(3, 3, intArrayOf(0, 2, 3, 3), intArrayOf(0, 2, 1), doubleArrayOf(2.0, 3.0, 4.0))

    @Test
    fun `every level two routine stages its alias from the workspace`() {
        val n = 4

        assertStaged(n, "gemv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.gemv(0.5, permutation(), y, -1.5, y, false, w)
        }
        assertStaged(n, "gemv over the coefficients it writes") { w ->
            val a = permutation()
            koblas.gemv(0.5, a, doubleArrayOf(1.0, -2.0, 0.5, 3.0), -1.5, a.values, false, w)
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.symv(0.5, permutation(), y, -1.5, y, true, w)
        }
        assertStaged(3, "trsv over the coefficients that are its right-hand side") { w ->
            val a = triangle()
            koblas.trsv(a, a.values, lower = true, workspace = w)
        }
        assertStaged(3, "trmv over the coefficients that are its right-hand side") { w ->
            val a = triangle()
            koblas.trmv(a, a.values, lower = true, workspace = w)
        }
    }

    /**
     * What a staged sparse operand costs, which is one loan rather than three: a structural array is never a
     * destination, so the snapshot shares the pointers and row indices and copies the coefficients alone.
     */
    @Test
    fun `a staged sparse operand copies its coefficients and shares its structure`() {
        val workspace = Workspace()
        val a = permutation()

        koblas.gemv(0.5, a, doubleArrayOf(1.0, -2.0, 0.5, 3.0), -1.5, a.values, false, workspace)

        assertEquals(1, workspace.available(a.nnz), "the coefficients were not staged from the workspace")
        assertEquals(0, workspace.availableI32(a.cols + 1), "the column pointers were copied")
        assertEquals(0, workspace.availableI32(a.nnz), "the row indices were copied")
    }

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Workspace) -> Unit) {
        val workspace = Workspace()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }
}
