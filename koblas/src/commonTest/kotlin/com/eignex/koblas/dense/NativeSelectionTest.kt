package com.eignex.koblas.dense

import com.eignex.koblas.internal.kernels.NativeContext
import com.eignex.koblas.internal.kernels.NativeGeometry
import com.eignex.koblas.internal.kernels.NativeHost
import com.eignex.koblas.internal.kernels.NativeKernel
import com.eignex.koblas.internal.kernels.NativeState
import com.eignex.koblas.internal.kernels.NativeTypes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class NativeSelectionTest {
    @Test
    fun `small and disabled host paths do not query current thread state`() {
        var queries = 0
        val kernel = syntheticKernel()
        val selection = NativeSelection(
            kernel,
            host(),
            HostCrossovers(WorkRule.Minimum(128), WorkRule.Never, WorkRule.AlwaysEligible),
        ) {
            queries++
            context()
        }
        assertNull(selection.auto(RuntimeCompetitor.JvmScalar, 127))
        assertNull(selection.auto(RuntimeCompetitor.JvmVector, Int.MAX_VALUE))
        assertEquals(0, queries)
        assertEquals(kernel.id, selection.auto(RuntimeCompetitor.JvmScalar, 128)?.id)
        assertEquals(1, queries)
    }

    @Test
    fun `exact selection bypasses performance rules but fails for unavailable capability`() {
        val disabled = HostCrossovers(WorkRule.Never, WorkRule.Never, WorkRule.Never)
        val kernel = syntheticKernel()
        assertEquals(kernel.id, NativeSelection(kernel, host(), disabled, ::context).exact().id)
        assertFailsWith<IllegalArgumentException> { NativeSelection(null, host(), disabled, ::context).exact() }
        val unavailable = NativeSelection(kernel, host().copy(usable = 0uL), disabled, ::context)
        assertFailsWith<IllegalArgumentException> { unavailable.exact() }
    }

    @Test
    fun `non Double matrix arithmetic remains ineligible even when enabled by tuning`() {
        val aceFeature = 1uL shl 10
        val ace = syntheticKernel().copy(requiredFeatures = aceFeature, types = NativeTypes(2, 2, 2, 2, 0))
        val aceHost = host().copy(hardware = aceFeature, usable = aceFeature, built = aceFeature)
        val choice = NativeSelection(ace, aceHost, always(), ::context)
        assertNull(choice.auto(RuntimeCompetitor.JvmScalar, 64, 64, 64))
        assertFailsWith<IllegalArgumentException> { choice.exact() }
    }

    @Test
    fun `future tile state uses its own process and thread requirements`() {
        val tile = syntheticKernel().copy(state = NativeState(4, 8, 4), executionMode = 0)
        val denied = NativeSelection(tile, host(), always(), ::context)
        assertNull(denied.auto(RuntimeCompetitor.JvmScalar, 32, 32, 32))
        assertFailsWith<IllegalArgumentException> { denied.exact() }
        val ready = NativeSelection(tile, host(), always()) { NativeContext(4, 8, null, null) }
        assertEquals(tile.id, ready.auto(RuntimeCompetitor.JvmVector, 32, 32, 32)?.id)
    }

    @Test
    fun `a wider alternative cannot replace the profile selected implementation`() {
        val narrow = syntheticKernel().copy(id = 102, geometry = geometry().copy(registerBits = 128))
        val wide = syntheticKernel().copy(id = 103, geometry = geometry().copy(registerBits = 512))
        val catalog = listOf(wide, narrow)
        val selected = NativeSelection(catalog.single { it.id == 102 }, host(), always(), ::context)
        assertEquals(narrow.id, selected.auto(RuntimeCompetitor.JvmScalar, 4096)?.id)
    }

    private fun always() = HostCrossovers(WorkRule.AlwaysEligible, WorkRule.AlwaysEligible, WorkRule.AlwaysEligible)
    private fun context() = NativeContext(0, 0, null, null)
    private fun host() = NativeHost(1, 1, 1uL shl 9, 1uL shl 9, 1uL shl 9, 0, 0, 0, 0, 0)
    private fun geometry() = NativeGeometry(
        256, 1, 4, 1, 1, 16, 8, 1,
        PackedMatrixLayout.GROUPED_FP64, PackedMatrixLayout.GROUPED_FP64, 0, 8, 0,
    )
    private fun syntheticKernel() = NativeKernel(
        100, 10, 99, 0, 1uL shl 9, NativeTypes(1, 1, 1, 1, 0),
        geometry(), NativeState(0, 0, 0), 0, 2, 7, 1, 3,
    )
}
