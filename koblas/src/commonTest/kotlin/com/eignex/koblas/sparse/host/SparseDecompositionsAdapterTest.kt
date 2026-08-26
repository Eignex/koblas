package com.eignex.koblas.sparse.host

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.backend.platformDispatchThresholds
import com.eignex.koblas.sparse.F64SparseFactorization
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SparseDecompositionsAdapterTest {

    /** An adapter that accelerates nothing and counts what the gates let through. */
    private class RecordingAdapter(factorizeMin: Int? = null) : F64SparseDecompositionsAdapter(factorizeMin) {
        var factors = 0
        var choleskys = 0
        var ldls = 0

        override val name: String get() = "recording"
        override val nativeAvailable: Boolean get() = true

        override fun factorNative(a: F64SparseMatrix): F64SparseFactorization {
            factors++
            return portable.factor(a)
        }

        override fun choleskyNative(a: F64SparseMatrix): F64SparseFactorization {
            choleskys++
            return portable.cholesky(a)
        }

        override fun ldlNative(a: F64SparseMatrix): F64SparseFactorization {
            ldls++
            return portable.ldl(a)
        }
    }

    /** A diagonal of [n] entries, whose stored count is [n] and which every factorization here accepts. */
    private fun diagonal(n: Int) = F64SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 2.0)) })

    @Test
    fun `the symmetric factorizations gate later than the general one`() {
        val thresholds = platformDispatchThresholds
        assertTrue(
            thresholds.symmetricFactorize > thresholds.factorize,
            "the symmetric gate should sit past the general one, got ${thresholds.symmetricFactorize} " +
                "against ${thresholds.factorize}",
        )
        val between = (thresholds.factorize + thresholds.symmetricFactorize) / 2
        val adapter = RecordingAdapter()

        adapter.factor(diagonal(between))
        adapter.cholesky(diagonal(between))
        adapter.ldl(diagonal(between))

        assertEquals(1, adapter.factors, "the general factorization is past its gate here")
        assertEquals(0, adapter.choleskys, "the Cholesky is not past its own gate here")
        assertEquals(0, adapter.ldls, "the LDL is not past its own gate here")
    }

    @Test
    fun `a backend naming its own gate moves all three`() {
        val adapter = RecordingAdapter(factorizeMin = 0)

        adapter.factor(diagonal(4))
        adapter.cholesky(diagonal(4))
        adapter.ldl(diagonal(4))

        assertEquals(1, adapter.factors)
        assertEquals(1, adapter.choleskys, "a caller asking for one gate means it for the symmetric ones too")
        assertEquals(1, adapter.ldls)
    }

    @Test
    fun `past the symmetric gate all three reach the library`() {
        val adapter = RecordingAdapter()
        val large = diagonal(platformDispatchThresholds.symmetricFactorize + 1)

        adapter.factor(large)
        adapter.cholesky(large)
        adapter.ldl(large)

        assertEquals(1, adapter.factors)
        assertEquals(1, adapter.choleskys)
        assertEquals(1, adapter.ldls)
    }
}
