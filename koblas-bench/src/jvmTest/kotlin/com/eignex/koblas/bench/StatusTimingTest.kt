package com.eignex.koblas.bench

import kotlin.test.*

class StatusTimingTest {
    private fun csv(case: String = "dot+64+uniform", value: String = "12.5", status: String = "ok",
                    samples: Int = 3, forks: Int = 1, provider: String = "scalar") =
        "$CSV_HEADER\n$case,$status,direct,arithmetic,scalar/dot,$provider,key,$samples,$forks,$value,$value,$value\n"

    @Test fun `status timings retain requested engines independently of scalar fallbacks`() {
        val case = Cases.parse("dot+64+uniform").single()

        for (engine in listOf("jvm-default", "jvm-scalar", "jvm-simd", "native", "native-default")) {
            val timing = readStatusTiming(engine, case, 3, csv())

            assertEquals(engine, timing.engine)
            assertEquals(12.5, timing.nanos)
            assertEquals("scalar", timing.provider)
            assertEquals("scalar/dot", timing.kernel)
            assertEquals(if (engine == "jvm-default") "median_ns" else "median_ns/$engine", timing.metric)
        }
    }

    @Test fun `status timings reject malformed or incomplete successful rows`() {
        val case = Cases.parse("dot+64+uniform").single()
        for (text in listOf(csv(value = "NaN"), csv(value = "0"), csv(samples = 2), csv(forks = 2),
            csv(provider = ""), csv(case = "dot+32+uniform"), csv() + csv().substringAfter('\n'))) {
            assertFailsWith<IllegalArgumentException> { readStatusTiming("jvm-default", case, 3, text) }
        }
    }

    @Test fun `only optional engines may decline known view adapters`() {
        val case = Cases.parse("copy-overlap+17+uniform").single()
        val unsupported = csv(case.id, value = "", status = "unsupported", samples = 0, forks = 0, provider = "")

        for (engine in listOf("jvm-scalar", "jvm-simd", "native")) {
            assertNull(readStatusTiming(engine, case, 3, unsupported).nanos)
        }
        for (engine in listOf("jvm-default", "native-default")) {
            assertFailsWith<IllegalArgumentException> { readStatusTiming(engine, case, 3, unsupported) }
        }
        assertFailsWith<IllegalArgumentException> {
            readStatusTiming("jvm-simd", Cases.parse("dot+64+uniform").single(), 3,
                csv(value = "", status = "unsupported", samples = 0, forks = 0))
        }
    }
}
