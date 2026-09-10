@file:Suppress("MatchingDeclarationName") // parsing and resolving the same JVM scatter setting

package com.eignex.koblas.sparse

import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.configuration.environmentVariableOrNull
import com.eignex.koblas.internal.configuration.systemPropertyOrNull

/** The requested use of indexed Vector API stores in sparse kernels. */
internal enum class JvmVectorScatterMode {
    AUTO,
    ON,
    OFF,
    ;

    companion object {
        fun configured(property: String?, environment: String?): JvmVectorScatterMode = when (
            (property ?: environment)?.trim()?.lowercase()
        ) {
            null, "", "auto" -> AUTO

            "on" -> ON

            "off" -> OFF

            else -> error(
                "koblas.jvm.vector.scatter must be auto, on, or off; got " +
                    "${property ?: environment}",
            )
        }
    }
}

/** The JVM Vector API indexed-store decision, resolved once with the sparse kernels. */
internal fun configuredJvmVectorScatter(): Boolean = jvmVectorScatterEnabled(
    JvmVectorScatterMode.configured(
        systemPropertyOrNull(VECTOR_SCATTER_PROPERTY),
        environmentVariableOrNull(VECTOR_SCATTER_ENVIRONMENT),
    ),
    simdAvailable,
    simdAvailable && SparseSimd.autoScatterEligible,
)

internal fun jvmVectorScatterEnabled(
    mode: JvmVectorScatterMode,
    vectorApiAvailable: Boolean,
    autoScatterEligible: Boolean,
): Boolean = when (mode) {
    JvmVectorScatterMode.OFF -> false

    JvmVectorScatterMode.ON -> {
        check(vectorApiAvailable) {
            "koblas.jvm.vector.scatter=on requires --add-modules=jdk.incubator.vector"
        }
        true
    }

    JvmVectorScatterMode.AUTO -> vectorApiAvailable && autoScatterEligible
}

private const val VECTOR_SCATTER_PROPERTY = "koblas.jvm.vector.scatter"
private const val VECTOR_SCATTER_ENVIRONMENT = "KOBLAS_JVM_VECTOR_SCATTER"
