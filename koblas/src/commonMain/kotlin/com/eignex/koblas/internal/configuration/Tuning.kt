package com.eignex.koblas.internal.configuration

/*
 * Reading one tuning entry, shared by the dense and sparse collections of them.
 *
 * The entries themselves live beside the routines they steer, in `DenseTuning` and `SparseTuning`, because
 * a block size means nothing without the measurement that chose it. What is common is only how a value is
 * found and what makes one unacceptable, which is here so the two collections cannot drift apart on either.
 */

/** The system property spelling of one tuning entry, for example `koblas.dense.level3.block.rows`. */
internal fun tuningProperty(prefix: String, name: String): String = "koblas.$prefix.$name"

/** The environment variable spelling of one tuning entry, for example `KOBLAS_DENSE_LEVEL3_BLOCK_ROWS`. */
internal fun tuningEnvironment(prefix: String, name: String): String =
    "KOBLAS_" + (prefix + "." + name).uppercase().replace('.', '_')

/** What a deployment configured for one entry, the system property ahead of the environment variable. */
internal fun configuredTuning(prefix: String, name: String): String? =
    systemPropertyOrNull(tuningProperty(prefix, name)) ?: environmentVariableOrNull(tuningEnvironment(prefix, name))

/** One whole-number entry, resolved once by its caller. */
internal fun tunedInt(
    prefix: String,
    name: String,
    default: Int,
    minimum: Int = 1,
    maximum: Int = Int.MAX_VALUE,
): Int = tunedIntValue(configuredTuning(prefix, name), default, minimum, maximum)

/** One fractional entry, resolved once by its caller. */
internal fun tunedDouble(prefix: String, name: String, default: Double, minimum: Double, maximum: Double): Double =
    tunedDoubleValue(configuredTuning(prefix, name), default, minimum, maximum)

/**
 * The whole number [configured] asks for, or [default] where it asks for nothing usable.
 *
 * Separate from reading the property so the decision can be tested without a process whose environment says
 * what a test needs it to say. Blank counts as unset: a variable exported empty clears an override.
 *
 * An override outside `[minimum, maximum]` is ignored rather than clamped or rejected loudly. That follows
 * how a configured library path that is not absolute is ignored: a typo in a deployment's environment must
 * not take a numerical library down while its classes load, and a block size of zero would do exactly that.
 */
internal fun tunedIntValue(configured: String?, default: Int, minimum: Int, maximum: Int): Int {
    val requested = configured?.trim()?.takeIf { it.isNotEmpty() }?.toIntOrNull() ?: return default
    return if (requested in minimum..maximum) requested else default
}

/** The fraction [configured] asks for, or [default], on the same terms as [tunedIntValue]. */
internal fun tunedDoubleValue(configured: String?, default: Double, minimum: Double, maximum: Double): Double {
    val requested = configured?.trim()?.takeIf { it.isNotEmpty() }?.toDoubleOrNull() ?: return default
    if (requested.isNaN()) return default
    return if (requested in minimum..maximum) requested else default
}
