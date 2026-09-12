package com.eignex.koblas.dense

import com.eignex.koblas.internal.configuration.configuredTuning

/** Startup-only resolution; callers freeze the results in their immutable runtime profile. */
internal class ProfileOverrides(private val configured: (String) -> String? = { configuredTuning("dense", it) }) {
    private val messages = mutableListOf<String>()
    val diagnostics: List<String> get() = messages.toList()

    fun rule(key: String, fallback: WorkRule): WorkRule {
        val text = configured(key)?.trim()?.takeIf { it.isNotEmpty() } ?: return fallback
        return when (text.lowercase()) {
            "never" -> WorkRule.Never

            "always" -> WorkRule.AlwaysEligible

            else -> {
                val minimum = text.toLongOrNull()
                if (minimum != null && minimum >= 0) {
                    WorkRule.Minimum(minimum)
                } else {
                    messages += "$key=$text: expected never, always, or nonnegative work; using conservative rule"
                    fallback
                }
            }
        }
    }

    fun schedule(fallback: BlockSchedule = BlockSchedule()): BlockSchedule {
        val rows = positiveInt("packed.block.rows", fallback.rows)
        val columns = positiveInt("packed.block.columns", fallback.columns)
        val depth = positiveInt("packed.block.depth", fallback.depth)
        val diagonal = positiveInt("triangular.block", fallback.diagonalBlock, Long.SIZE_BITS)
        val rhs = positiveInt("triangular.rhs.block", fallback.rhsBlock)
        val work = positiveLong("native.call.work.limit", fallback.nativeCallWorkLimit)
        return try {
            fallback.copy(
                rows = rows,
                columns = columns,
                depth = depth,
                diagonalBlock = diagonal,
                rhsBlock = rhs,
                nativeCallWorkLimit = work,
            )
        } catch (_: IllegalArgumentException) {
            messages += "block schedule exceeds storage capacity; using conservative schedule"
            fallback
        }
    }

    private fun positiveInt(key: String, fallback: Int, maximum: Int = Int.MAX_VALUE): Int {
        val text = configured(key)?.trim()?.takeIf { it.isNotEmpty() } ?: return fallback
        val value = text.toIntOrNull()
        return if (value != null && value in 1..maximum) {
            value
        } else {
            messages += "$key=$text: expected 1..$maximum; using $fallback"
            fallback
        }
    }

    private fun positiveLong(key: String, fallback: Long): Long {
        val text = configured(key)?.trim()?.takeIf { it.isNotEmpty() } ?: return fallback
        val value = text.toLongOrNull()
        return if (value != null && value > 0) {
            value
        } else {
            messages += "$key=$text: expected positive work; using $fallback"
            fallback
        }
    }
}
