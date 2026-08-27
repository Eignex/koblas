// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.test.Test
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
        val factorization = assertNotNull(cholmod.factorLdl(a))

        assertFailsWith<SingularMatrix> { factorization.l }
        assertFailsWith<SingularMatrix> { factorization.d }
        assertFailsWith<SingularMatrix> { factorization.order }
    }
}
