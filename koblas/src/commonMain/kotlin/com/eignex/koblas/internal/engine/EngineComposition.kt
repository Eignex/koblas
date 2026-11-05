package com.eignex.koblas.internal.engine

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.engine.EngineProfile
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.koblas

/**
 * Combines available arithmetic with ordered routing preferences and complete portable coverage.
 *
 * Backends named by no applicable rule contribute no arithmetic. Known unavailable built-in backends fall through
 * to later preferences and then portable Kotlin; unknown source identifiers are rejected. This never changes
 * [koblas]. The caller chooses an applicable profile explicitly; automatic platform selection also checks
 * [EngineProfile.appliesTo]. Panel, product and triangular families are selected unconditionally before a
 * schedule starts, because packing and grouping must use the same geometry as the selected arithmetic.
 * Whole-call portable preferences retain the shared schedule with those separately selected families.
 */
internal fun combine(profile: EngineProfile, vararg backends: KoblasEngine): KoblasEngine {
    require(profile.rules.none { rule -> !rule.unconditional && rule.implementations.any { it.family } }) {
        "Scheduled kernel families require unconditional selection before packing"
    }
    // Every engine this library builds supplies the kernel seam, which composition binds.
    return CombinedEngine(CompiledRouting(profile, backends.map { it as EngineKernels }), profile.name)
}
