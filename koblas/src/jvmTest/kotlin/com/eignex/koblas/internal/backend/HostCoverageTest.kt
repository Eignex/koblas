package com.eignex.koblas.internal.backend

import com.eignex.koblas.sparse.host.SparseBackends
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.experimental.categories.Category
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Every other host test skips itself when its library is missing, so a run on a machine with none of them
 * passes while covering nothing. This one fails instead, the way the native host suite already requires its
 * library rather than skipping. It runs only under `-Pkoblas.hostTests=true`, which is a caller saying the
 * libraries are there.
 */
@Category(HostLibraryTest::class)
class HostCoverageTest {

    @Test
    fun `an opt-in host run reaches at least one host library`() {
        val resolved = resolvedHostLibraries()
        assertTrue(
            resolved.isNotEmpty(),
            "HFactor did not resolve, so this run covered none of the bindings it was asked to exercise. " +
                "Install HFactor or drop -Pkoblas.hostTests=true.",
        )
    }

    /** Without this, a detection that reported a library unconditionally would make the guard above useless. */
    @Test
    fun `nothing resolves when every path is pointed at a library that is not there`() {
        val nowhere = "/nonexistent/koblas-host-coverage"
        assertEquals(
            emptyList(),
            resolvedHostLibraries(HfactorConfig(nowhere)),
        )
    }

    /** Reads the same configuration discovery does, so the guard reports what a run will actually reach. */
    private fun resolvedHostLibraries(
        hfactor: HfactorConfig = HfactorConfig(libraryPath(ConfigurationKeys.HFACTOR_PATH)),
    ): List<String> {
        val sparse = SparseBackends(hfactor)
        return buildList {
            if (sparse.hfactor.isAvailable) add("hfactor")
        }
    }
}
