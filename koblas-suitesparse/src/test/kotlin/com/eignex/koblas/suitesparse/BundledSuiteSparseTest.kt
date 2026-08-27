package com.eignex.koblas.suitesparse

import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull

class BundledSuiteSparseTest {

    /**
     * Every backend in this artifact asks for the whole manifest, since a library's dependencies ship beside
     * it and the loader wants them in one directory. Extracting per backend would lay the same bundle down
     * once per binding, and there are more bindings to come than there are libraries in it.
     */
    @Test
    fun `the bundled backends share one extraction of the artifact`() {
        val klu = assertNotNull(BundledKlu().config.libraryPath, "the bundled KLU resolved no library")
        val umfpack = assertNotNull(BundledUmfpack().config.libraryPath, "the bundled UMFPACK resolved no library")

        assertEquals(
            Path.of(klu).parent,
            Path.of(umfpack).parent,
            "the artifact was extracted once per backend rather than once",
        )
    }

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
