package com.eignex.koblas.suitesparse

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull

class BundledSuiteSparseTest {

    @Test
    fun `the artifact contains no unreachable SPQR payload`() {
        val library = assertNotNull(BundledKlu().config.libraryPath, "the bundled KLU resolved no library")
        val names = Path.of(library).parent.resolve(".libraries").toFile().readLines()
        val notices = assertNotNull(javaClass.classLoader.getResourceAsStream("THIRD-PARTY-NOTICES.txt"))
            .bufferedReader().use { it.readText() }

        assertFalse(names.any { "spqr" in it.lowercase() })
        assertFalse("SPQR" in notices)
    }
}
