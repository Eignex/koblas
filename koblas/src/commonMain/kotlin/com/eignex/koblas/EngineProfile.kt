package com.eignex.koblas

import com.eignex.koblas.vendor.*

/** One positive routing preference. Earlier matching rules take precedence over later ones. */
public class EngineRule(
    /** Identifier of a backend supplying the requested implementations. */
    public val backend: String,
    implementations: Set<EngineImplementation>,
    /** Minimum run length or matrix row extent. */
    public val rows: Int = 0,
    /** Maximum run length or matrix row extent covered by measurements. */
    public val maximumRows: Int = Int.MAX_VALUE,
    /** Minimum output column extent. */
    public val columns: Int = 0,
    /** Maximum output column extent covered by measurements. */
    public val maximumColumns: Int = Int.MAX_VALUE,
    /** Minimum shared dimension or structured order. */
    public val depth: Int = 0,
    /** Maximum shared dimension supported by this measured preference. */
    public val maximumDepth: Int = Int.MAX_VALUE,
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
        require(rows >= 0 && columns >= 0 && depth >= 0 && maximumDepth >= depth && work >= 0)
        require(maximumRows >= rows && maximumColumns >= columns)
        require(this.implementations.isNotEmpty())
    }

    internal val unconditional: Boolean get() = rows == 0 && columns == 0 && depth == 0 &&
        maximumRows == Int.MAX_VALUE && maximumColumns == Int.MAX_VALUE &&
        maximumDepth == Int.MAX_VALUE && work == 0L && contiguous == null && borrowed == null

    internal fun matches(r: Int, c: Int, d: Int, adjacent: Boolean, view: Boolean): Boolean {
        if (r < rows || r > maximumRows || c < columns || c > maximumColumns || d < depth || d > maximumDepth ||
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
    /** Execution runtime, such as `jvm` or `native`, or null for any runtime. */
    public val runtime: String? = null,
    /** Complete measured host, or null for any host. */
    public val target: HostPlatform? = null,
    /** Retained measurement location or a description of the configuration's provenance. */
    public val evidence: String = "explicit configuration",
) {
    /** Ordered preferences, copied at construction so caller mutations cannot change an engine. */
    public val rules: List<EngineRule> = rules.toList()

    /** Whether this profile was declared applicable to the supplied runtime and platform. */
    public fun appliesTo(runtime: String, host: HostPlatform): Boolean =
        (this.runtime == null || this.runtime == runtime) &&
            (
                target == null || (
                    target.operatingSystem == host.operatingSystem &&
                        target.architecture == host.architecture && target.cpuVendor == host.cpuVendor
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
            require(header.size == 8 && header[0] == "koblas-profile-v1") { "Invalid profile header" }
            fun optional(value: String): String? = value.takeUnless { it == "*" }
            fun flag(value: String): Boolean? = if (value == "*") null else value.toBooleanStrict()
            val rules = lines.drop(1).map { fields ->
                require(fields.size == 11) { "Invalid routing rule" }
                EngineRule(
                    fields[0], fields[1].split(',').map(EngineImplementation::valueOf).toSet(),
                    rows = fields[2].toInt(), maximumRows = fields[3].toInt(),
                    columns = fields[4].toInt(), maximumColumns = fields[5].toInt(),
                    depth = fields[6].toInt(), maximumDepth = fields[7].toInt(), work = fields[8].toLong(),
                    contiguous = flag(fields[9]), borrowed = flag(fields[10]),
                )
            }
            return EngineProfile(
                header[1],
                rules,
                optional(header[2]),
                if (header.subList(3, 6).all { it == "*" }) {
                    null
                } else {
                    HostPlatform(
                        OperatingSystem.valueOf(header[3]),
                        Architecture.valueOf(header[4]),
                        CpuVendor.valueOf(header[5]),
                    )
                },
                header[6] + "; " + header[7],
            )
        }
    }
}
