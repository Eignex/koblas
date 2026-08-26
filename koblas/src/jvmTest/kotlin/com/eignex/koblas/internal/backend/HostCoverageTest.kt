package com.eignex.koblas.internal.backend

import com.eignex.koblas.dense.host.cblas.HostBlasConfig
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.sparse.host.F64SparseBackends
import com.eignex.koblas.sparse.host.basiclu.BasicluConfig
import com.eignex.koblas.sparse.host.hfactor.HfactorConfig
import com.eignex.koblas.sparse.host.klu.KluConfig
import com.eignex.koblas.sparse.host.umfpack.UmfpackConfig
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
            "no host library resolved, so this run covered none of the bindings it was asked to exercise. " +
                "Install OpenBLAS, SuiteSparse or BASICLU, or drop -Pkoblas.hostTests=true.",
        )
    }

    /** Without this, a detection that reported a library unconditionally would make the guard above useless. */
    @Test
    fun `nothing resolves when every path is pointed at a library that is not there`() {
        val nowhere = "/nonexistent/koblas-host-coverage"
        assertEquals(
            emptyList(),
            resolvedHostLibraries(
                blas = HostBlasConfig(libraryPath = nowhere, lapackeLibraryPath = nowhere),
                klu = KluConfig(nowhere),
                umfpack = UmfpackConfig(libraryPath = nowhere),
                basiclu = BasicluConfig(nowhere),
                hfactor = HfactorConfig(nowhere),
            ),
        )
    }

    /** Reads the same configuration discovery does, so the guard reports what a run will actually reach. */
    private fun resolvedHostLibraries(
        blas: HostBlasConfig = HostBlasConfig(
            libraryPath = libraryPath(ConfigurationKeys.CBLAS_PATH),
            lapackeLibraryPath = libraryPath(ConfigurationKeys.LAPACKE_PATH),
        ),
        klu: KluConfig = KluConfig(libraryPath(ConfigurationKeys.KLU_PATH)),
        umfpack: UmfpackConfig = UmfpackConfig(libraryPath = libraryPath(ConfigurationKeys.UMFPACK_PATH)),
        basiclu: BasicluConfig = BasicluConfig(libraryPath(ConfigurationKeys.BASICLU_PATH)),
        hfactor: HfactorConfig = HfactorConfig(libraryPath(ConfigurationKeys.HFACTOR_PATH)),
    ): List<String> {
        val dense = HostBlasCalls(blas)
        val sparse = F64SparseBackends(klu, umfpack, basiclu, hfactor)
        return buildList {
            if (dense.available) add("cblas")
            if (dense.lapackAvailable) add("lapacke")
            if (sparse.klu.isAvailable) add("klu")
            if (sparse.umfpack.isAvailable) add("umfpack")
            if (sparse.basiclu.isAvailable) add("basiclu")
            if (sparse.hfactor.isAvailable) add("hfactor")
        }
    }
}
