package com.eignex.koblas.sparse.internal

import com.eignex.koblas.Workspace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SparseOwnershipTest {
    @Test
    fun `symmetric rank scratch returns every buffer after failure`() {
        val workspace = Workspace()

        assertFailsWith<ExpectedFailure> {
            withSymmetricRankScratch(workspace, order = 5, sourceRows = 7, sourceEntries = 11) { _,
                    _, _, _, _, _, _,
                ->
                throw ExpectedFailure()
            }
        }

        assertEquals(1, workspace.pooledWidths)
        assertEquals(4, workspace.pooledI32Widths)
    }

    private class ExpectedFailure : RuntimeException()
}
