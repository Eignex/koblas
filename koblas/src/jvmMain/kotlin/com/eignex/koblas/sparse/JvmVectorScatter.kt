package com.eignex.koblas.sparse

import com.eignex.koblas.dense.simdAvailable
import com.eignex.koblas.internal.configuration.ConfigurationKeys
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
internal class JvmVectorScatter private constructor(val mode: JvmVectorScatterMode, val enabled: Boolean) {
    val path: String get() = if (enabled) "indexed-store" else "scalar"

    val options: Map<String, String>
        get() = mapOf(
            "jvm.vector.scatter.mode" to mode.name.lowercase(),
            "jvm.vector.scatter.path" to path,
        )

    companion object {
        fun configured(): JvmVectorScatter {
            val mode = JvmVectorScatterMode.configured(
                systemPropertyOrNull(ConfigurationKeys.JVM_VECTOR_SCATTER.property),
                environmentVariableOrNull(ConfigurationKeys.JVM_VECTOR_SCATTER.environment),
            )
            return resolve(mode, simdAvailable, simdAvailable && SparseSimd.autoScatterEligible)
        }

        fun resolve(
            mode: JvmVectorScatterMode,
            vectorApiAvailable: Boolean,
            autoScatterEligible: Boolean,
        ): JvmVectorScatter {
            val enabled = when (mode) {
                JvmVectorScatterMode.OFF -> false

                JvmVectorScatterMode.ON -> {
                    check(vectorApiAvailable) {
                        "koblas.jvm.vector.scatter=on requires --add-modules=jdk.incubator.vector"
                    }
                    true
                }

                JvmVectorScatterMode.AUTO -> vectorApiAvailable && autoScatterEligible
            }
            return JvmVectorScatter(mode, enabled)
        }
    }
}
