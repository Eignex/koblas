package com.eignex.koblas.bench

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.sparse.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RunnerTest {
    @Test
    fun `recorded providers follow the actual threshold and layout dispatcher`() {
        var calls = 0
        val leaf = object : KoblasEngine {
            override val name = "chosen"
            override val implementations = setOf(EngineImplementation.Dot)
            override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int, aStride: Int, bStride: Int): Double {
                calls++
                return BuiltinEngines.scalar.dot(a, aOff, b, bOff, len, aStride, bStride)
            }
        }
        val engine = KoblasEngine.combine(EngineProfile("threshold", listOf(
            EngineRule(leaf.id, leaf.implementations, rows = 64..Int.MAX_VALUE, contiguous = true),
        )), leaf)
        for ((id, expected) in listOf("dot+63+uniform" to "portable", "dot+64+uniform" to leaf.id,
            "dot+64+uniform+stride=2" to "portable")) {
            calls = 0
            val work = engineArm(Cases.parse(id).single(), engine)!!.work!!
            work.run()
            assertEquals(expected, work.provider)
            assertEquals(expected == leaf.id, calls > 0)
        }
    }

    @Test
    fun `runner rejects malformed numeric options instead of using defaults`() {
        for (option in listOf("warmups", "samples", "target-ms", "forks")) {
            for (value in listOf("", "many", "1.5", "999999999999999999999999")) {
                assertFailsWith<IllegalArgumentException>("$option=$value") {
                    parseArguments(arrayOf("--mode=native", "--$option=$value"))
                }
            }
        }
    }

    @Test
    fun `runner accepts an explicit targeted suite`() {
        val sweep = parseArguments(
            arrayOf("--mode=native", "--suite=sweep", "--operation=dot", "--warmups=0", "--samples=1", "--target-ms=1"),
        )

        assertEquals("sweep", sweep.suite)
        assertEquals("dot", sweep.operation)
        assertEquals(1_000_000L, sweep.targetNanos)
    }

    @Test
    fun `runner uses the default suite when none is supplied`() {
        val settings = parseArguments(arrayOf("--mode=jvm-scalar"))

        assertEquals("default", settings.suite)
    }

    @Test
    fun `runner rejects invalid suite selections`() {
        for (args in listOf(arrayOf("--suite=sweep"), arrayOf("--suite=other"), arrayOf("--suite=default", "--suite=sweep"))) {
            assertFailsWith<IllegalArgumentException> { parseArguments(arrayOf("--mode=native") + args) }
        }
    }

    @Test
    fun `suite metadata does not change report comparison fields`() {
        for (id in listOf("dot+4096+uniform", "gemm+15x7x31+uniform")) {
            for (mode in listOf("jvm-scalar", "jvm-simd", "native")) {
                val original = Cases.parse(id).single()
                val shared = Cases.parse("$id+suite=default,sweep").single()
                val settings = settings().copy(mode = mode)
                val before = measurement(original, settings, 1, 2.5, "ok", "direct", "arithmetic", "scalar/dot", "portable")
                val after = measurement(
                    shared,
                    settings.copy(suite = "sweep"),
                    1,
                    2.5,
                    "ok",
                    "direct",
                    "arithmetic",
                    "scalar/dot", "portable",
                )

                assertEquals(reportCsv(listOf(before)), reportCsv(listOf(after)))
            }
        }
    }

    @Test
    fun `report summarizes samples across forks without run metadata`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurements = listOf(
            measurement(case, settings(), 1, 2.5, "ok", "direct", "arithmetic", "scalar/dot", "portable"),
            measurement(case, settings(), 2, 3.5, "ok", "direct", "arithmetic", "scalar/dot", "portable"),
        )

        val records = reportCsv(measurements).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,ok,direct,arithmetic,scalar/dot,portable,Dot:4:1:1:false:1:true,2,2,3.0,2.5,3.5"), records)
    }

    @Test
    fun `unsupported case has no timing`() {
        val case = Cases.parse("dot+4+uniform").single()
        val measurement = measurement(case, settings(), 0, null, "unsupported", "unsupported", "arithmetic")

        val records = reportCsv(listOf(measurement)).trim().lines()

        assertEquals(listOf(CSV_HEADER, "dot+4+uniform,unsupported,unsupported,arithmetic,unavailable,,Dot:4:1:1:false:1:true,0,0,,,"), records)
    }

    /** A timed row that cannot say which call produced it is a bug, not a row with a missing field. */
    @Test
    fun `a timed record without attribution is refused rather than reported`() {
        val case = Cases.parse("dot+4+uniform").single()

        val failure = assertFailsWith<IllegalStateException> {
            measurement(case, settings(), 1, 2.5, "ok", "direct", "arithmetic")
        }

        assertEquals(true, "carries no attribution" in failure.message.orEmpty(), failure.message)
    }

    @Test
    fun `reports carry the routing dimensions used by the engine`() {
        for ((id, key) in listOf(
            "copy+32+uniform+stride=-2" to "Copy:32:1:1:false:1:false",
            "spdot-sparse+64+sparse-uniform+density=0.1" to "IndexedSparseDot:6:1:1:false:1:true",
            "spdot-slice+64+sparse-uniform+density=0.25" to "IndexedDot:16:1:1:false:1:true",
            "spnrm2-indexed+64+sparse-uniform+density=0.1" to "IndexedNrm2:6:1:1:false:1:true",
            "scal+32+uniform" to "Scale:32:1:1:false:1:true",
            "gemv+16x32+uniform" to "Gemv:16:32:1:false:2:true",
            "gemv+16x32+uniform+transA=T" to "GemvTransposed:32:16:1:false:2:true",
            "trsm+16x32+triangular+side=R+uplo=L+transA=N+diag=N" to "Trsm:16:32:32:false:3:true",
            "syrk+16x32+uniform+uplo=L+transA=T" to "Syrk:16:16:32:false:3:true",
            "gemm+16x32x64+uniform+layout=borrowed" to "Gemm:16:32:64:true:3:true",
        )) {
            val row = measurement(Cases.parse(id).single(), settings(), 0, null, "unsupported", "unsupported", "arithmetic")

            assertEquals(key, row.case.last())
        }
    }

    private fun settings() = Settings("jvm-scalar", "all", "cases.txt", "output.csv", 0, 1, 1_000_000, 2)
}
