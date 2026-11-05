@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.engine.engineFromSetting
import com.eignex.koblas.internal.engine.platformEngineSetting

private var selectedEngine: KoblasEngine = engineFromSetting(platformEngineSetting())

/**
 * The shared BLAS engine used by convenience operations and libraries using this Koblas instance.
 *
 * Initialized from `koblas.engine` on JVM, then `KOBLAS_ENGINE`, or automatic platform selection.
 * [configureKoblas] can override that initial value during startup. Callers must finish configuration
 * before any operation uses this engine and safely publish it before concurrent use.
 * Reads and configuration are unsynchronized; the library does not enforce startup order.
 */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine get() = selectedEngine

/**
 * Chooses [engine] for [koblas] during application startup.
 *
 * Call before libraries use Koblas and before starting concurrent work. This directly assigns the shared
 * engine without synchronization or lifecycle checks. The platform or launch-selected default initializes
 * before this assignment, so launch settings must be valid even when overridden programmatically.
 *
 * [engine] is used as supplied, including any unsupported operations of an exact vendor engine from
 * [KoblasEngine]. Direct calls to other engine instances are unaffected.
 */
public fun configureKoblas(engine: KoblasEngine) {
    selectedEngine = engine
}

/**
 * Selects [koblas] by the same name as the launch setting: `auto`, `scalar`, `simd`, or a vendor name.
 *
 * Call during ordered startup, before libraries use Koblas or concurrent work begins. Names ignore case
 * and surrounding whitespace. Unknown or unavailable selections fail in this call; vendor selections retain
 * portable coverage for unexported operations. Initial launch settings must also be valid.
 */
public fun configureKoblas(engine: String) {
    selectedEngine = engineFromSetting(engine)
}
