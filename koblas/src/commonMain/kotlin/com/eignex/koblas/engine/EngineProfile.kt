package com.eignex.koblas.engine

import com.eignex.koblas.*

/** Execution bindings with distinct routing costs. */
public enum class EngineRuntime {
    /** JVM arrays and optional foreign-memory transfers. */
    Jvm,

    /** Kotlin/Native arrays and pinned foreign calls. */
    Native,
}

/** One positive routing preference. Earlier matching rules take precedence over later ones. */
public class EngineRule(
    /** Identifier of a backend supplying the requested implementations. */
    public val source: String,
    implementations: Set<EngineImplementation>,
    /** Covered run lengths or matrix row extents. */
    public val rows: IntRange = 0..Int.MAX_VALUE,
    /** Covered output column extents. */
    public val columns: IntRange = 0..Int.MAX_VALUE,
    /** Covered shared dimensions or structured orders. */
    public val depth: IntRange = 0..Int.MAX_VALUE,
    /** Minimum rows times columns times depth, compared without overflowing. */
    public val work: Long = 0,
    /** Required vector layout, or null for either layout. */
    public val contiguous: Boolean? = null,
    /** Required matrix ownership/layout, or null for either layout. */
    public val borrowed: Boolean? = null,
) {
    /** Implementations selected together by this rule. */
    public val implementations: Set<EngineImplementation> = implementations.toSet()

    init {
        require(listOf(rows, columns, depth).all { it.first >= 0 && !it.isEmpty() })
        require(work >= 0)
        require(this.implementations.isNotEmpty())
    }

    internal val unconditional: Boolean get() = rows == 0..Int.MAX_VALUE && columns == 0..Int.MAX_VALUE &&
        depth == 0..Int.MAX_VALUE && work == 0L && contiguous == null && borrowed == null

    internal fun matches(r: Int, c: Int, d: Int, adjacent: Boolean, view: Boolean): Boolean {
        if (r !in rows || c !in columns || d !in depth ||
            (contiguous != null && contiguous != adjacent) || (borrowed != null && borrowed != view)
        ) {
            return false
        }
        if (work == 0L) return true
        if (r <= 0 || c <= 0 || d <= 0) return false
        val area = r.toLong() * c
        return area >= work / d + if (work % d == 0L) 0 else 1
    }
}

/** A reproducible routing configuration. Missing or inapplicable preferences retain portable coverage. */
public class EngineProfile(
    /** Stable profile identifier, including its revision. */
    public val name: String,
    rules: List<EngineRule>,
    /** Execution binding, or null for either runtime. */
    public val runtime: EngineRuntime? = null,
    /** Target OS, architecture and optional CPU model and instruction class, or null for any host. */
    public val target: HostPlatform? = null,
    /** Retained measurement location or a description of the configuration's provenance. */
    public val evidence: String = "explicit configuration",
) {
    /** Ordered preferences, copied at construction so caller mutations cannot change an engine. */
    public val rules: List<EngineRule> = rules.toList()

    init {
        require(
            target?.cpuClass != CpuClass.Sve || target.sveVectorBits != null,
        ) { "SVE profiles require a measured vector width" }
    }

    /** Whether this profile was declared applicable to the supplied runtime and platform. */
    public fun appliesTo(runtime: EngineRuntime, host: HostPlatform): Boolean =
        (this.runtime == null || this.runtime == runtime) &&
            (
                target == null || (
                    target.operatingSystem == host.operatingSystem &&
                        target.architecture == host.architecture &&
                        (target.cpuModel == null || target.cpuModel == host.cpuModel) &&
                        (target.cpuClass == null || target.cpuClass == host.cpuClass) &&
                        (target.sveVectorBits == null || target.sveVectorBits == host.sveVectorBits)
                    )
                )
}
