package com.eignex.koblas.internal.kernels

import com.eignex.koblas.NativeVariant

/** Read-only C ABI transport. Both runtimes decode the same native-order words. */
internal expect object NativeProbe {
    fun query(kind: Int, index: Int = 0, kernelId: Int = 0): IntArray?
}

/** Hardware facts are distinct from compiled support and process or thread readiness. */
internal data class NativeHost(
    val architecture: Int,
    val operatingSystem: Int,
    val hardware: ULong,
    val usable: ULong,
    val built: ULong,
    val vendor: Int,
    val family: Int,
    val model: Int,
    val tuningKey: Int,
    val validity: Int,
)

/** Operation-specific numerical types; a matrix accelerator is not necessarily an FP64 implementation. */
internal data class NativeTypes(val a: Int, val b: Int, val accumulation: Int, val output: Int, val mode: Int) {
    val isDouble: Boolean get() = a == 1 && b == 1 && accumulation == 1 && output == 1 && mode == 0
}

/** Register width, logical grouping, and matrix layout are independent facts. */
internal data class NativeGeometry(
    val registerBits: Int,
    val widthMode: Int,
    val logicalBatch: Int,
    val unroll: Int,
    val accumulators: Int,
    val rows: Int,
    val columns: Int,
    val depthMultiple: Int,
    val leftLayout: Int,
    val rightLayout: Int,
    val outputLayout: Int,
    val alignment: Int,
    val scratchBytes: Int,
)

/** Process permission, thread preparation and per-call ABI state have different lifetimes. */
internal data class NativeState(val process: Int, val thread: Int, val call: Int)

/** Immutable, operation-specific description. Unknown tags stay representable and are never executed. */
internal data class NativeKernel(
    val id: Int,
    val operation: Int,
    val variant: Int,
    val reason: Int,
    val requiredFeatures: ULong,
    val types: NativeTypes,
    val geometry: NativeGeometry,
    val state: NativeState,
    val executionMode: Int,
    val primitive: Int,
    val tails: Int,
    val semantics: Int,
    val addressing: Int,
)

/** Null vector lengths mean unknown or unavailable, distinguished by the host's feature facts. */
internal data class NativeContext(
    val readyProcess: Int,
    val readyThread: Int,
    val ordinaryVectorBytes: Int?,
    val streamingVectorBytes: Int?,
)

/** Decoder for the sized v1 record, shared by JVM and Native. */
internal object NativeRecords {
    fun validate(words: IntArray): IntArray {
        require(
            words.size >= 64 && words[0] == 1 && words[1] >= 256 &&
                words[1].toLong() <= words.size.toLong() * 4 && words[3] == 0,
        ) {
            "invalid native probe record"
        }
        return words
    }

    private fun IntArray.features(offset: Int): ULong =
        this[offset].toUInt().toULong() or (this[offset + 1].toUInt().toULong() shl 32)

    fun host(words: IntArray): NativeHost = validate(words).let {
        NativeHost(
            it[6], it[7],
            it.features(
                12,
            ),
            it.features(14), it.features(16), it[8], it[9], it[10], it[55], it[11],
        )
    }

    fun kernel(words: IntArray): NativeKernel = validate(words).let {
        NativeKernel(
            it[18], it[19], it[20], it[21], it.features(22),
            NativeTypes(it[24], it[25], it[26], it[27], it[28]),
            NativeGeometry(
                it[31],
                it[30],
                it[32],
                it[33],
                it[34],
                it[36],
                it[37],
                it[38],
                it[40],
                it[41],
                it[42],
                it[43],
                it[44],
            ),
            NativeState(it[47], it[48], it[49]), it[29], it[35], it[39], it[45], it[46],
        )
    }

    fun context(words: IntArray): NativeContext = validate(words).let {
        NativeContext(
            it[50],
            it[51],
            it[52].takeIf { _ -> it[54] and 1 != 0 },
            it[53].takeIf { _ -> it[54] and 2 != 0 },
        )
    }
}

/** Type, feature and state eligibility only; callers separately validate operation, layout and shape.
 * Synthetic records can inspect future requirements without authorizing any real C entry point.
 */
internal fun nativeEligibility(kernel: NativeKernel, host: NativeHost, context: NativeContext): Int = when {
    kernel.reason != 0 -> kernel.reason
    !kernel.types.isDouble -> 8
    kernel.requiredFeatures and host.built != kernel.requiredFeatures -> 9
    kernel.requiredFeatures and host.hardware != kernel.requiredFeatures -> 5
    kernel.requiredFeatures and host.usable != kernel.requiredFeatures -> 6
    kernel.state.process and context.readyProcess != kernel.state.process -> 7
    kernel.state.thread and context.readyThread != kernel.state.thread -> 8
    else -> 0
}

/** The C catalog is immutable; current-thread information is always queried separately. */
internal object NativeCatalog {
    private val hostWords = NativeProbe.query(1)?.let(NativeRecords::validate)
    val host: NativeHost? = hostWords?.let(NativeRecords::host)
    val kernels: List<NativeKernel> = hostWords?.let { words ->
        List(words[5]) { index -> NativeRecords.kernel(checkNotNull(NativeProbe.query(2, index))) }
    }.orEmpty()
    val variants: List<NativeVariant> = NativeVariant.entries.filter { variant ->
        kernels.any { it.variant == variant.id } && kernels.filter { it.variant == variant.id }.all { it.reason == 0 }
    }

    // Preserve the former ordinary clone choice until measured operation policy migrates in W05/W26.
    val defaultVariant: NativeVariant? = listOf(
        NativeVariant.AVX2,
        NativeVariant.NEON,
        NativeVariant.SSE2,
        NativeVariant.SCALAR,
    )
        .firstOrNull { it in variants }

    fun requireVariant(variant: NativeVariant) {
        require(
            variant in variants,
        ) { "native ${variant.name} is unavailable: ${kernels.firstOrNull { it.variant == variant.id }?.reason ?: 9}" }
    }
}
