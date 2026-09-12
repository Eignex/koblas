package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SparseSlicesWorkTest {
    @Test
    fun `reuse cycles agree with the scalar reference and preserve output tails`() {
        for (checked in listOf(false, true)) for (compact in listOf(false, true)) {
            for (locality in listOf("sorted", "shuffled")) for (exceptional in listOf(false, true)) {
                val operation = if (checked) "cycle-checked" else "cycle"
                val work = create(operation, "+compact=${if (compact) "T" else "N"}+locality=$locality")
                if (exceptional) {
                    val values = doubleArrayOf(0.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY,
                        Double.MIN_VALUE, -Double.MIN_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE, 1.0)
                    values.copyInto(work.values)
                }
                val expectedAccumulator = DoubleArray(31) { if (it in work.indices) 0.0 else 19.0 }
                val expectedMarks = IntArray(31) { if (it in work.indices) 0 else 7 }
                expectedAccumulator.copyInto(work.accumulator)
                expectedMarks.copyInto(work.marks)
                work.outIndices.fill(-1); work.outValues.fill(23.0)
                val expectedIndices = IntArray(work.count) { -1 }
                val expectedValues = DoubleArray(work.count) { 23.0 }
                var expectedCount = 0
                for (k in 0 until work.count) {
                    val first = if (k < (work.count + 1) / 2) 0.0 + 0.875 * work.values[k] else 0.0
                    val value = first + -0.875 * work.values[k]
                    if (!compact || value != 0.0) {
                        expectedIndices[expectedCount] = work.indices[k]
                        expectedValues[expectedCount++] = value
                    }
                }

                repeat(3) {
                    work.run()

                    assertEquals(expectedCount, work.written)
                    assertContentEquals(expectedIndices, work.outIndices)
                    assertContentEquals(expectedValues, work.outValues)
                    assertContentEquals(expectedAccumulator, work.accumulator)
                    assertContentEquals(expectedMarks, work.marks)
                    assertContentEquals(work.indices, work.touched)
                    assertEquals(if (checked && exceptional) 1 else 0, work.status[0])
                }
            }
        }
    }

    @Test
    fun `gather and clear comparisons preserve entries outside touched support`() {
        for (operation in listOf("gather", "gather-clear", "clear")) {
            val work = create(operation, "+locality=shuffled")
            val expected = DoubleArray(31) { 19.0 }
            val expectedMarks = IntArray(31) { 7 }
            for (k in work.indices.indices) {
                val index = work.indices[k]
                work.accumulator[index] = work.values[k]
                work.marks[index] = 1
                expected[index] = if (operation == "gather") work.values[k] else 0.0
                expectedMarks[index] = if (operation == "gather") 1 else 0
            }
            for (index in expected.indices) if (index !in work.indices) {
                work.accumulator[index] = 19.0; work.marks[index] = 7
            }

            repeat(3) {
                work.run()

                assertContentEquals(expected, work.accumulator)
                assertContentEquals(expectedMarks, work.marks)
                if (operation != "clear") {
                    assertContentEquals(work.indices, work.outIndices)
                    assertContentEquals(work.values, work.outValues)
                }
            }
        }
    }

    @Test
    fun `checked dot comparison retains ordered cancellation and status consumption`() {
        val work = create("reduce-dot-checked")
        work.values.fill(1.0)
        for (index in work.indices) work.accumulator[index] = 0.0
        work.accumulator[work.indices[0]] = 1e16
        work.accumulator[work.indices[1]] = 1.0
        work.accumulator[work.indices[2]] = -1e16

        assertEquals(0.0, work.run())
        assertEquals(0, work.status[0])

        work.values.fill(Double.MIN_VALUE)
        for (index in work.indices) work.accumulator[index] = Double.MIN_VALUE
        assertEquals(2.0, work.run())
        assertEquals(2, work.status[0])
    }

    @Test
    fun `shuffled fixture matches the independent C producer`() {
        val work = create("cycle", "+locality=shuffled")

        assertContentEquals(intArrayOf(23, 19, 16, 21, 12, 9, 10, 15, 2), work.indices)
        assertEquals("82a2f9d0ba820fb8", digest(work.values))
    }

    @Test
    fun `reuse comparisons have distinct timing and explicit portable kernel metadata`() {
        val case = Cases.parse("sparse-slices-gather+31+sparse-uniform+density=0.3+timing=reuse").single()
        val legacy = Cases.parse("sparse-slices-gather+31+sparse-uniform+density=0.3").single()
        val current = sparseWork(case, BuiltinEngines.scalar)!!
        val old = sparseWork(legacy, BuiltinEngines.scalar)!!

        assertEquals("composed", current.comparisonKind)
        assertEquals("slices-gather-v1", current.timingMode)
        assertEquals("sparse-slices", old.timingMode)
        val settings = Settings("jvm-simd", "all", "cases.txt", "output.csv", 0, 1, 1_000_000, 1, "1", "source", "false")
        val row = measurement(case, "jvm-simd", settings, 1, 1, 1, "1", "ok", current.comparisonKind, current.timingMode, "test")
        assertEquals("portable-sparse-slices", row.case.last())
    }

    @Test
    fun `comparison options require the reuse timing contract`() {
        for (text in listOf(
            "sparse-slices-cycle+64+sparse-uniform+density=0.25",
            "sparse-slices-gather+64+sparse-uniform+density=0.25+timing=arithmetic",
            "sparse-slices-clear+64+sparse-uniform+density=0.25+timing=reuse+compact=T",
            "sparse-slices-gather+64+sparse-uniform+density=0.25+locality=shuffled",
            "sparse-slices-cycle+64+sparse-uniform+density=0.25+timing=reuse+locality=random",
        )) assertFailsWith<IllegalArgumentException> { Cases.parse(text) }
    }

    private fun create(operation: String, options: String = ""): SparseSlicesReuseWork {
        val case = Cases.parse("sparse-slices-$operation+31+sparse-uniform+density=0.3+timing=reuse$options").single()
        return SparseSlicesReuseWork(case, BuiltinEngines.scalar)
    }
}
