package com.eignex.koblas

import com.eignex.koblas.vendor.*

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

    /** Loading configurations does no filesystem access and never changes the process default. */
    public companion object {
        /**
         * Reads the versioned tab-separated format emitted by `koblas-bench/tune-engine.py`.
         * Unknown versions, malformed fields and invalid bounds are rejected before an engine is built.
         */
        public fun parse(text: String): EngineProfile {
            val lines = text.lineSequence().filter { it.isNotBlank() && !it.startsWith('#') }
                .map { it.split('\t') }.toList()
            require(lines.isNotEmpty()) { "Empty profile" }
            val header = lines.first()
            require(header.size == 10 && header[0] == "koblas-profile-v3") { "Invalid profile header" }
            fun optional(value: String): String? = value.takeUnless { it == "*" }
            fun flag(value: String): Boolean? = if (value == "*") null else value.toBooleanStrict()
            val rules = lines.drop(1).map { fields ->
                require(fields.size == 11) { "Invalid routing rule" }
                EngineRule(
                    fields[0],
                    fields[1].split(',').map(EngineImplementation::valueOf).toSet(),
                    rows = fields[2].toInt()..fields[3].toInt(),
                    columns = fields[4].toInt()..fields[5].toInt(),
                    depth = fields[6].toInt()..fields[7].toInt(),
                    work = fields[8].toLong(),
                    contiguous = flag(fields[9]),
                    borrowed = flag(fields[10]),
                )
            }
            return EngineProfile(
                header[1],
                rules,
                optional(header[2])?.let { EngineRuntime.valueOf(it.replaceFirstChar(Char::uppercase)) },
                if (header.subList(3, 8).all { it == "*" }) {
                    null
                } else {
                    HostPlatform(
                        OperatingSystem.valueOf(header[3]),
                        Architecture.valueOf(header[4]),
                        optional(header[5]),
                        optional(header[6])?.let(CpuClass::valueOf),
                        optional(header[7])?.toInt(),
                    )
                },
                header[8] + "; " + header[9],
            )
        }
    }
}
