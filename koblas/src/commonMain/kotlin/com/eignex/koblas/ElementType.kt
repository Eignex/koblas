package com.eignex.koblas

/**
 * An element type koblas keys things by: the dispatch thresholds and the configuration keys that override
 * them. It is not what picks a container or a backend half, since those carry their element type in their
 * own names ([F64DenseMatrix], [com.eignex.koblas.dense.F64Blas]); it is for the machinery that is written
 * once and then asked which element type it is answering for.
 *
 * [F64] is the only one implemented today. The other two are declared here so that what is keyed by element
 * type has a settled key before a second element type arrives, rather than the double-precision key being
 * the bare one and every later key being invented on the spot.
 *
 * @property keySegment what this element type adds to a system property name. Empty for [F64], so the
 *   double-precision keys are the unqualified ones, as the double-precision type names are.
 * @property envSegment the same for an environment variable name.
 */
internal enum class ElementType(val keySegment: String, val envSegment: String) {
    /** `Double` storage and arithmetic. */
    F64("", ""),

    /** `Float` storage and arithmetic. */
    F32("f32.", "F32_"),

    /** `Short` storage of bfloat16, accumulating in `Float`. */
    BF16("bf16.", "BF16_"),
}
