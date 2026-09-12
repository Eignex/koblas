package com.eignex.koblas.dense

import com.eignex.koblas.NativeVariant

/** Resolved Kotlin policy. Construction snapshots entries; no tuning source is read by an operation call. */
internal class DenseProfile(
    crossovers: Map<DenseOperation, HostCrossovers>,
    val schedule: BlockSchedule,
    val products: ProductPolicy,
    diagnostics: List<String>,
) {
    private val entries = crossovers.toMap()
    val diagnostics: List<String> = diagnostics.toList()

    init {
        require(entries.keys == DenseOperation.entries.toSet()) { "incomplete dense crossover profile" }
    }

    operator fun get(operation: DenseOperation): HostCrossovers = entries.getValue(operation)
}

/**
 * Existing ordinary host-call choices, retained as conservative transition data. These are not calibrated
 * profiles for new widths or matrix accelerators. SIMD-to-C remains disabled until separately measured.
 */
internal object DenseProfiles {
    val conservative: DenseProfile by lazy { resolve(ProfileOverrides()) }

    // This preserves the former ordinary implementation choice; new variants require measured profile entries.
    fun preferredVariant(available: List<NativeVariant>): NativeVariant? =
        listOf(NativeVariant.AVX2, NativeVariant.NEON, NativeVariant.SSE2, NativeVariant.SCALAR)
            .firstOrNull { it in available }

    fun resolve(overrides: ProfileOverrides): DenseProfile {
        val entries = DenseOperation.entries.associateWith { operation ->
            HostCrossovers(
                overrides.rule("jvm.c.${operation.key}.crossover", scalarDefault(operation)),
                overrides.rule("jvm.simd.c.${operation.key}.crossover", WorkRule.Never),
                overrides.rule("native.c.${operation.key}.crossover", nativeDefault(operation)),
            )
        }
        val schedule = overrides.schedule()
        return DenseProfile(entries, schedule, ProductPolicy(), overrides.diagnostics)
    }

    private fun scalarDefault(operation: DenseOperation): WorkRule = when (operation) {
        DenseOperation.Dot, DenseOperation.Sum, DenseOperation.Nrm2, DenseOperation.Asum ->
            WorkRule.Minimum(128)

        DenseOperation.Ssqd, DenseOperation.DotAxpy -> WorkRule.Minimum(256)

        DenseOperation.Iamax -> WorkRule.Minimum(4096)

        DenseOperation.Dot4 -> WorkRule.Minimum(512)

        DenseOperation.Axpy4 -> WorkRule.Minimum(64)

        DenseOperation.GemmTile, DenseOperation.GemmTrsmTile -> WorkRule.Minimum(16)

        else -> WorkRule.Never
    }

    private fun nativeDefault(operation: DenseOperation): WorkRule = when (operation) {
        DenseOperation.Dot, DenseOperation.Axpy, DenseOperation.Scale, DenseOperation.Iamax,
        DenseOperation.Axpy4, DenseOperation.DotAxpy, DenseOperation.AxpyArithmetic,
        ->
            WorkRule.Minimum(48)

        else -> WorkRule.AlwaysEligible
    }
}
