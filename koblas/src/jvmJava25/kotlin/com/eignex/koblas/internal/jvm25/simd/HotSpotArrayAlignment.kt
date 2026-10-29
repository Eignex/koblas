package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import com.sun.management.HotSpotDiagnosticMXBean
import jdk.incubator.vector.DoubleVector
import java.lang.management.ManagementFactory

/**
 * A checked-array loop can peel to a vector boundary when HotSpot guarantees its object's alignment,
 * when G1 puts a humongous array at a region start, or for ZGC medium and large objects.
 * These use the array header layout checked here.
 * Unknown layouts keep their ordinary loop; the hint never licenses unchecked access.
 */
internal object HotSpotArrayAlignment {
    private val ALIGNMENT_BYTES = DoubleVector.SPECIES_PREFERRED.vectorByteSize()
    private val HEADER_BYTES = headerBytes()
    private val OBJECT_ALIGNMENT_BYTES = objectAlignmentBytes()
    private val REGION_BYTES = regionBytes()
    private val ZGC_ALIGNED_ARRAYS = zgcAlignedArrays()

    fun prefix(values: DoubleArray, offset: Int, length: Int): Int {
        if (!hasKnownAlignment(values)) return 0
        val bytes = (-(HEADER_BYTES + offset * 8L) and (ALIGNMENT_BYTES - 1).toLong()).toInt()
        return minOf(length, bytes / 8)
    }

    private fun headerBytes(): Int = runCatching {
        val feature = Runtime.version().feature()
        if ((feature != 25 && feature != 27) || System.getProperty("sun.arch.data.model") != "64") {
            return@runCatching 0
        }
        // JDK 27 always compresses class pointers; the old flag is no longer queryable.
        if (feature == 27) return@runCatching 16
        val bean = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java)
        val compressed = bean.getVMOption("UseCompressedClassPointers").value == "true"
        val compact = bean.getVMOption("UseCompactObjectHeaders").value == "true"
        if (compressed || compact) 16 else 24
    }.getOrDefault(0)

    private fun objectAlignmentBytes(): Int = runCatching {
        if (HEADER_BYTES == 0) return@runCatching 0
        val bean = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java)
        val bytes = bean.getVMOption("ObjectAlignmentInBytes").value.toInt()
        if (bytes >= ALIGNMENT_BYTES && bytes % ALIGNMENT_BYTES == 0) bytes else 0
    }.getOrDefault(0)

    private fun regionBytes(): Long = runCatching {
        if (HEADER_BYTES == 0 || ManagementFactory.getGarbageCollectorMXBeans().none { it.name.startsWith("G1 ") }) {
            return@runCatching 0L
        }
        val bean = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean::class.java)
        val bytes = bean.getVMOption("G1HeapRegionSize").value.toLong()
        if (bytes >= ALIGNMENT_BYTES && bytes % ALIGNMENT_BYTES == 0L) bytes else 0L
    }.getOrDefault(0L)

    private fun zgcAlignedArrays(): Boolean = HEADER_BYTES != 0 && ALIGNMENT_BYTES <= 512 &&
        ManagementFactory.getGarbageCollectorMXBeans().any { it.name.startsWith("ZGC ") }

    /** Whether this runtime and array geometry determine alignment at a vector boundary. */
    fun hasKnownAlignment(values: DoubleArray): Boolean = OBJECT_ALIGNMENT_BYTES != 0 ||
        (REGION_BYTES != 0L && HEADER_BYTES + values.size * 8L > REGION_BYTES / 2) ||
        // On the checked JDKs, ZGC objects above 256 KiB have at least 512-byte object alignment.
        (ZGC_ALIGNED_ARRAYS && HEADER_BYTES + values.size * 8L > 256 * 1024)
}
