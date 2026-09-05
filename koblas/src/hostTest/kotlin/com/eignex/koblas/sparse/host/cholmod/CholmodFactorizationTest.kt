// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull

/** Checks factor access that depends on CHOLMOD's native factor state. */
class CholmodFactorizationTest {
    private val cholmod = CholmodCholesky().also {
        require(it.isAvailable) { "host SuiteSparse expected in the test environment" }
    }

    @Test
    fun `a singular LDL factorization has no factors to give`() {
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), emptyList()))
        val factorization = assertNotNull(cholmod.factorQuasiDefiniteLdl(a))

        assertFailsWith<SingularMatrix> { factorization.l }
        assertFailsWith<SingularMatrix> { factorization.d }
        assertFailsWith<SingularMatrix> { factorization.order }
    }

    @Test
    fun `repeated factorizations release the common they start`() {
        // Each native factorization starts its own cholmod_common, and analyze and factorize fill in Flag,
        // Head, Iwork and Xwork inside it. Freeing the block without cholmod_finish leaked all of that,
        // roughly 40 bytes per row every time round. The loop is what makes a regression show as growth
        // rather than as a single small leak.
        val a = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 4.0, 1 to 1.0), listOf(1 to 3.0, 2 to 1.0), listOf(2 to 2.0)),
        )

        repeat(200) {
            val factorization = assertNotNull(cholmod.factor(a))
            assertEquals(3, factorization.n)
            factorization.close()
        }
    }

    @Test
    fun `a matrix CHOLMOD cannot factor is not reported as one it factored`() {
        // A non-positive pivot returns TRUE from cholmod_factorize and shows up in `minor`, which is what
        // this must keep raising. The return check added beside it only rejects a FALSE, so this pins that
        // the two stay apart.
        val indefinite = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 4.0, 2 to 3.0), listOf(1 to 4.0), listOf(2 to 2.0)),
        )

        assertFailsWith<NotPositiveDefinite> { cholmod.factor(indefinite) }
    }
}
