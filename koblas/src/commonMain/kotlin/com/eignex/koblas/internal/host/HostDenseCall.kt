package com.eignex.koblas.internal.host

import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation

/** One typed host selection from the same compiled preferences as vector and panel arithmetic. */
internal fun interface DenseHostSelection {
    fun select(operation: EngineImplementation, rows: Int, columns: Int, depth: Int, borrowed: Boolean): Blas?
}

/** Capabilities follow actual BLAS exports; transpose variants share their underlying symbol. */
internal object HostDenseCall {
    private val exports = EngineImplementation.entries.map { operation ->
        val name = if (operation == EngineImplementation.Scale) "Scal" else operation.name.removeSuffix("Transposed")
        BlasOperation.entries.firstOrNull { it.name == name }
    }

    fun entryPointFor(operation: EngineImplementation): BlasOperation? = exports[operation.ordinal]
}
