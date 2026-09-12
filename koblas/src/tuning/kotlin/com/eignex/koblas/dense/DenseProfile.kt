package com.eignex.koblas.dense

/** Operation policy keys; native operation identities remain in the native catalog. */
internal enum class DensePolicyOperation(val key: String) {
    Dot("dot"),
    Sum("sum"),
    Ssqd("ssqd"),
    Nrm2("nrm2"),
    Iamax("iamax"),
    Asum("asum"),
    Axpy("axpy"),
    Scale("scale"),
    Swap("swap"),
    Rot("rot"),
    Rotm("rotm"),
    Dot4("dot4"),
    Axpy4("axpy4"),
    DotAxpy("dot.axpy"),
    AxpyArithmetic("axpy.arithmetic"),
    GemmTile("gemm.tile"),
    TrsmTile("trsm.tile"),
    GemmTrsmTile("gemm.trsm.tile"),
}

/** Resolved Kotlin policy. Construction snapshots entries; no tuning source is read by an operation call. */
internal class DenseProfile(
    crossovers: Map<DensePolicyOperation, HostCrossovers>,
    val schedule: BlockSchedule,
    val products: ProductPolicy,
    diagnostics: List<String>,
) {
    private val entries = crossovers.toMap()
    val diagnostics: List<String> = diagnostics.toList()

    init {
        require(entries.keys == DensePolicyOperation.entries.toSet()) { "incomplete dense crossover profile" }
    }

    operator fun get(operation: DensePolicyOperation): HostCrossovers = entries.getValue(operation)
}

/**
 * Existing ordinary host-call choices, retained as conservative transition data. These are not calibrated
 * profiles for new widths or matrix accelerators. SIMD-to-C remains disabled until separately measured.
 */
internal object DenseProfiles {
    val conservative: DenseProfile by lazy { resolve(ProfileOverrides()) }

    fun resolve(overrides: ProfileOverrides): DenseProfile {
        val entries = DensePolicyOperation.entries.associateWith { operation ->
            HostCrossovers(
                overrides.rule("jvm.c.${operation.key}.crossover", scalarDefault(operation)),
                overrides.rule("jvm.simd.c.${operation.key}.crossover", WorkRule.Never),
                overrides.rule("native.c.${operation.key}.crossover", nativeDefault(operation)),
            )
        }
        val schedule = overrides.schedule()
        return DenseProfile(entries, schedule, ProductPolicy(), overrides.diagnostics)
    }

    private fun scalarDefault(operation: DensePolicyOperation): WorkRule = when (operation) {
        DensePolicyOperation.Dot, DensePolicyOperation.Sum, DensePolicyOperation.Nrm2, DensePolicyOperation.Asum ->
            WorkRule.Minimum(128)

        DensePolicyOperation.Ssqd, DensePolicyOperation.DotAxpy -> WorkRule.Minimum(256)

        DensePolicyOperation.Iamax -> WorkRule.Minimum(4096)

        DensePolicyOperation.Dot4 -> WorkRule.Minimum(512)

        DensePolicyOperation.Axpy4 -> WorkRule.Minimum(64)

        DensePolicyOperation.GemmTile, DensePolicyOperation.GemmTrsmTile -> WorkRule.Minimum(16)

        else -> WorkRule.Never
    }

    private fun nativeDefault(operation: DensePolicyOperation): WorkRule = when (operation) {
        DensePolicyOperation.Dot, DensePolicyOperation.Axpy, DensePolicyOperation.Scale, DensePolicyOperation.Iamax,
        DensePolicyOperation.Axpy4, DensePolicyOperation.DotAxpy, DensePolicyOperation.AxpyArithmetic,
        ->
            WorkRule.Minimum(48)

        else -> WorkRule.AlwaysEligible
    }
}
