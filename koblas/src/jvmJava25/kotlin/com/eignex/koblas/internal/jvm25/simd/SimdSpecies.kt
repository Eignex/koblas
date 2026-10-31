package com.eignex.koblas.internal.jvm25.simd

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorShape
import jdk.incubator.vector.VectorSpecies

/**
 * The one double species every Vector API kernel here uses.
 *
 * The preferred species, except on a 512-bit x86 unit, where it is the 256-bit one. A Java array's elements
 * start 16 bytes past a cache line and the JVM cannot be asked to align them, so every 64-byte access splits
 * across two lines while only every other 32-byte one does; on the measured AVX-512 hosts the narrower vectors
 * were faster in every dense Level 2 routine. A 128-bit unit never splits and keeps its own width.
 *
 * Every kernel must share it: a second species reaching the Vector API's shared entry points deoptimises them,
 * and kernels that never used it start allocating. Reading it initialises the Vector API, so only code that has
 * already found [simdAvailable] may.
 */
internal val SIMD_SPECIES: VectorSpecies<Double> = DoubleVector.SPECIES_PREFERRED.let { preferred ->
    val bits = simdBitSizeFor(System.getProperty("os.arch").orEmpty(), preferred.vectorBitSize())
    if (bits == preferred.vectorBitSize()) preferred else preferred.withShape(VectorShape.forBitSize(bits))
}

/** The vector width [SIMD_SPECIES] takes on architecture [osArch] whose preferred width is [preferredBits]. */
internal fun simdBitSizeFor(osArch: String, preferredBits: Int): Int =
    if (osArch in setOf("amd64", "x86_64", "x64") && preferredBits > 256) 256 else preferredBits
