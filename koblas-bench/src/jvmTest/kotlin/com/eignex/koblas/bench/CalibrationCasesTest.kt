package com.eignex.koblas.bench

import kotlin.test.*

class CalibrationCasesTest {
    @Test fun `generated grids are unique disjoint and parseable`() {
        val training = CalibrationCases.training(true)
        val holdout = CalibrationCases.holdout(true)

        assertEquals(training.size, training.toSet().size)
        assertEquals(holdout.size, holdout.toSet().size)
        assertTrue(training.toSet().intersect(holdout.toSet()).isEmpty())
        Cases.parse((training + holdout).joinToString("\n"))
    }
}
