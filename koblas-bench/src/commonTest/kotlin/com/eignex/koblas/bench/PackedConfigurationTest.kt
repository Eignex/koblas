package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class PackedConfigurationTest {
    @Test
    fun `every packed configuration field is required`() {
        val fields = BLOCK.split('+')
        for (field in fields.drop(3)) {
            assertFailsWith<IllegalArgumentException>(field) { Cases.parse(fields.filterNot { it == field }.joinToString("+")) }
        }
    }

    @Test
    fun `unsupported layout and schedule overrides are rejected`() {
        for ((from, to) in listOf("alignment=8" to "alignment=64", "leftStride=4" to "leftStride=8",
            "panel=31" to "panel=32", "batch=1" to "batch=2", "leftLayout=depth-rows-v1" to "leftLayout=depth-rows-v2",
            "variant=current-tile-v1" to "variant=sme", "timing=prepacked-compute" to "timing=raw-tile")) {
            assertFailsWith<IllegalArgumentException>(to) { Cases.parse(BLOCK.replace(from, to)) }
        }
    }

    @Test
    fun `logical identity is independent of tile geometry`() {
        val four = Cases.parse(BLOCK).single()
        val eight = Cases.parse(eightRows()).single()

        assertEquals(four.logicalId, eight.logicalId)
        assertTrue(four.configurationId != eight.configurationId)
        assertTrue(four.physicalWork != eight.physicalWork)
    }

    @Test
    fun `logical matrices have stable digests before packing`() {
        assertEquals(digest(Fixtures.vector(15 * 31, 1)), digest(Fixtures.matrix(15, 31, 1).data))
        assertEquals(digest(Fixtures.vector(31 * 7, 2)), digest(Fixtures.matrix(31, 7, 2).data))
    }

    @Test
    fun `fixed work rejects a backend with different geometry`() {
        val case = Cases.parse(eightRows()).single()
        val before = case.physicalWork

        assertNull(denseWork(case, BuiltinEngines.scalar))
        assertEquals(before, case.physicalWork)
        assertEquals(16 * 31, PackedConfiguration(case).leftSize)
    }

    @Test
    fun `packers preserve the versioned formulas and positive zero padding`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.c, BuiltinEngines.simd).distinct()) {
            val specification = if (engine.packedKernels.gemmTileRows == 4) BLOCK else eightRows()
            val p = PackedConfiguration(Cases.parse(specification).single())
            val a = Fixtures.matrix(p.m, p.depth, 1)
            val b = Fixtures.matrix(p.depth, p.n, 2)
            a[0, 0] = -0.0
            b[0, 0] = -0.0
            val left = DoubleArray(p.leftSize + 2) { Double.NaN }
            val right = DoubleArray(p.rightSize + 2) { Double.NaN }

            engine.packedPanels.packLeft(a, left, p.m, p.depth, sourceRow = 0, sourceColumn = 0,
                transpose = false, alpha = 1.0, destinationOffset = 0, workspace = null)
            engine.packedPanels.packRight(b, right, p.depth, p.n, sourceRow = 0, sourceColumn = 0,
                transpose = false, destinationOffset = 0, workspace = null)

            for (tile in 0 until p.rowTiles) for (depth in 0 until p.depth) for (lane in 0 until p.rows) {
                val row = tile * p.rows + lane
                val expected = if (row < p.m) a[row, depth] else 0.0
                assertEquals(expected.toBits(), left[tile * p.rows * p.depth + depth * p.rows + lane].toBits())
            }
            for (tile in 0 until p.columnTiles) for (depth in 0 until p.depth) for (lane in 0 until p.columns) {
                val column = tile * p.columns + lane
                val expected = if (column < p.n) b[depth, column] else 0.0
                assertEquals(expected.toBits(), right[tile * p.columns * p.depth + depth * p.columns + lane].toBits())
            }
            for (index in p.leftSize until left.size) assertEquals(Double.NaN.toBits(), left[index].toBits())
            for (index in p.rightSize until right.size) assertEquals(Double.NaN.toBits(), right[index].toBits())
        }
    }

    @Test
    fun `block loops and packing modes agree with the scalar reference`() {
        val a = Fixtures.matrix(15, 31, 1)
        val b = Fixtures.matrix(31, 7, 2)
        val expected = Fixtures.matrix(15, 7, 4)
        BuiltinEngines.scalar.gemm(1.0, a, false, b, false, 1.0, expected)
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.c, BuiltinEngines.simd).distinct()) {
            val specification = if (engine.packedKernels.gemmTileRows == 4) BLOCK else eightRows()
            for (timing in listOf("prepacked-compute", "pack-plus-compute")) {
                val work = denseWork(Cases.parse(specification.replace("prepacked-compute", timing)).single(), engine) ?: continue
                repeat(2) {
                    work.run()
                    val actual = requireNotNull(work.result)
                    for (index in actual.indices) assertBlockAgreesWithReference(expected.data[index], actual[index])
                }
            }
        }
    }

    private fun assertBlockAgreesWithReference(expected: Double, actual: Double) {
        assertTrue(abs(expected - actual) < 2e-12 * (1.0 + abs(expected)), "$expected != $actual")
    }

    private fun eightRows(): String = BLOCK.replace("physical=4x4", "physical=8x4")
        .replace("leftGroup=4", "leftGroup=8").replace("leftStride=4", "leftStride=8")

    private companion object {
        const val BLOCK = "gemm-block+15x7x31+uniform+physical=4x4+work=gemm-add-v1+leftLayout=depth-rows-v1+rightLayout=depth-columns-v1+leftGroup=4+rightGroup=4+leftStride=4+rightStride=4+padding=zero+alignment=8+block=15x7x31+panel=31+diagonal=0+rhs=0+batch=1+variant=current-tile-v1+timing=prepacked-compute"
    }
}
