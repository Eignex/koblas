@file:OptIn(kotlin.concurrent.atomics.ExperimentalAtomicApi::class)

package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import kotlin.concurrent.atomics.AtomicReference

internal val defaultEngine: DefaultEngine = DefaultEngine()

/** Seals startup configuration before resolving providers, including when resolution fails. */
internal class DefaultEngine(private val resolve: () -> KoblasEngine = { engineFromSetting(platformEngineSetting()) }) {
    private sealed interface Selection {
        data object Automatic : Selection
        class Explicit(val create: () -> KoblasEngine) : Selection
        data object Started : Selection
    }

    private val selection = AtomicReference<Selection>(Selection.Automatic)

    // Cache a failure too: later calls must not silently select a different engine after startup failed.
    private val resolved: Result<KoblasEngine> by lazy {
        runCatching {
            when (val choice = selection.exchange(Selection.Started)) {
                Selection.Automatic -> resolve()
                is Selection.Explicit -> choice.create()
                Selection.Started -> error("Recursive Koblas engine selection")
            }
        }
    }

    val engine: KoblasEngine get() = resolved.getOrThrow()

    fun configure(engine: KoblasEngine): Unit = configure { engine }

    fun configure(create: () -> KoblasEngine) {
        check(selection.compareAndSet(Selection.Automatic, Selection.Explicit(create))) {
            "Koblas is already configured or engine selection has started; call configureKoblas once before first use"
        }
    }
}

/** Explicit names select only the requested source, with portable coverage for missing vendor operations. */
internal fun engineFromSetting(
    setting: String?,
    automatic: () -> KoblasEngine = ::platformEngine,
    simd: () -> KoblasEngine? = { BuiltinEngines.simd },
    openVendor: (Vendor) -> KoblasEngine? = BuiltinEngines::vendor,
): KoblasEngine {
    val name = setting?.trim()?.lowercase() ?: "auto"
    return when (name) {
        "auto" -> automatic()

        "scalar" -> BuiltinEngines.scalar

        "simd" -> checkNotNull(simd()) { "Koblas engine 'simd' is unavailable; enable the JVM Vector API module" }

        else -> {
            val vendor = requireNotNull(Vendor.entries.firstOrNull { it.vendorName.lowercase() == name }) {
                "Unknown Koblas engine '$setting'; expected auto, scalar, simd, " +
                    Vendor.entries.joinToString { it.vendorName.lowercase() }
            }
            val engine = checkNotNull(openVendor(vendor)) { "Koblas engine '$name' is unavailable on this platform" }
            combine(EngineTuning(name, listOf(EngineRule(engine.id, engine.implementations))), engine)
        }
    }
}
