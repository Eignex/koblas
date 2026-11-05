package com.eignex.koblas.bench

import com.eignex.koblas.KoblasEngine
import com.eignex.lab.bench.BenchRequest
import java.lang.reflect.Modifier
import kotlin.test.*

class StatusSweepTest {
    private val parameters = mapOf("status-sweep" to "true", "samples" to "3", "warmups" to "3",
        "target-ms" to "100", "forks" to "1", "rounds" to "1")

    @Test fun `fixed sweep covers every engine overload at two sizes and executes numerical checks`() {
        val methods = KoblasEngine::class.java.declaredMethods.filter {
            !Modifier.isStatic(it.modifiers) && !it.name.startsWith("get") && '$' !in it.name
        }
        val cases = KoblasLabSuite.plan(BenchRequest(parameters))
        assertEquals(41, methods.size, "Review sweep membership when the engine API changes")
        assertEquals(methods.size * 2, cases.size)
        val workloads = cases.map { Cases.parse(it.parameters.getValue("row")).single() }
        assertEquals(41, workloads.map { it.operation }.distinct().size)
        assertTrue(workloads.groupBy { it.operation }.values.all { it.size == 2 && it.map { c -> c.dimensions }.distinct().size == 2 })
        for (case in cases) {
            assertEquals("jvm-default", case.parameters["targets"])
            assertEquals("metric-v1", case.attributes["regressionPolicy"])
            assertEquals(60L, case.resources.timeoutSeconds)
            assertTrue(case.inputs.none { it.role == "executable" })
        }
        for (workload in workloads) {
            val work = assertNotNull(engineArm(workload, KoblasEngine.scalar)?.work, workload.id)
            try { assertTrue(work.run().isFinite(), workload.id) } finally { work.close() }
        }
    }

    @Test fun `status sweep refuses controls that would mix measurement series`() {
        for (override in listOf(mapOf("rounds" to "2"), mapOf("samples" to "1"), mapOf("targets" to "jvm-scalar"),
            mapOf("smoke" to "true"), mapOf("cases" to "koblas-bench/cases.txt"))) {
            assertFailsWith<IllegalArgumentException> { KoblasLabSuite.plan(BenchRequest(parameters + override)) }
        }
    }

}
