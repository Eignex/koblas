@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.engine.defaultEngine

/**
 * The immutable BLAS engine used by convenience operations and libraries sharing this Koblas instance.
 *
 * Selected once on first use. [configureKoblas] overrides launch settings; otherwise `koblas.engine` on
 * JVM takes precedence over `KOBLAS_ENGINE`. Without a setting, selection is automatic for the platform.
 */
@get:kotlin.jvm.JvmName("getDefault")
public val koblas: KoblasEngine by lazy { defaultEngine.engine }

/**
 * Chooses [engine] for [koblas] before its first use, overriding launch settings.
 *
 * Call once at application startup, before a library or convenience operation accesses [koblas]. The engine
 * is used as supplied, including any unsupported operations of an exact vendor engine from [BuiltinEngines].
 * Direct calls to other engine instances are unaffected.
 *
 * @throws IllegalStateException if already configured or if default engine selection has started.
 */
public fun configureKoblas(engine: KoblasEngine): Unit = defaultEngine.configure(engine)
