package com.eignex.koblas.bench

import com.sun.management.OperatingSystemMXBean
import java.lang.management.ManagementFactory
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.Locale

/** Samples host CPU usage and writes one summary per observed benchmark phase. */
public object CpuSampler {
    @JvmStatic
    public fun main(args: Array<String>) {
        val output = Path.of(args[0])
        val ready = Path.of(args[1])
        val stop = Path.of(args[2])
        val phaseFile = Path.of(args[3])
        val parent = ProcessHandle.current().parent().orElseThrow()
        val cpu = ManagementFactory.getOperatingSystemMXBean() as? OperatingSystemMXBean
        Files.newBufferedWriter(output).use { writer ->
            writer.appendLine("phase,start_s,end_s,readings,available_readings,mean_cpu_percent,median_cpu_percent,min_cpu_percent,max_cpu_percent")
            // Discard the first reading because its observation window is undefined.
            cpu?.cpuLoad
            val start = System.nanoTime()
            println("started_at=${Instant.now()}")
            println("runtime=${System.getProperty("java.vendor")}/${System.getProperty("java.version")}")
            var phase = Files.readString(phaseFile).trim()
            var phaseStart = 0.0
            var usage = CpuUsage()
            var samples = 0
            while (parent.isAlive && !Files.exists(stop)) {
                Thread.sleep(1000)
                val elapsed = (System.nanoTime() - start) / 1e9
                val nextPhase = Files.readString(phaseFile).trim()
                if (nextPhase != phase) {
                    writer.appendLine(usage.row(phase, phaseStart, elapsed))
                    writer.flush()
                    phase = nextPhase
                    phaseStart = elapsed
                    usage = CpuUsage()
                }
                usage.add(cpu?.cpuLoad ?: -1.0)
                if (++samples == 3) Files.createFile(ready)
            }
            writer.appendLine(usage.row(phase, phaseStart, (System.nanoTime() - start) / 1e9))
        }
    }
}

internal class CpuUsage {
    private var readings = 0
    private val values = arrayListOf<Double>()

    fun add(load: Double) {
        readings++
        if (load.isFinite() && load in 0.0..1.0) values += load * 100
    }

    fun row(phase: String, start: Double, end: Double): String {
        require(phase in setOf("baseline", "jvm-scalar", "jvm-c", "jvm-simd", "native", "vendors"))
        val sorted = values.sorted()
        val statistics = if (sorted.isEmpty()) listOf("", "", "", "") else {
            val middle = sorted.size / 2
            val median = if (sorted.size % 2 == 1) sorted[middle] else sorted[middle - 1] / 2 + sorted[middle] / 2
            listOf(sorted.average(), median, sorted.first(), sorted.last()).map { String.format(Locale.ROOT, "%.6g", it) }
        }
        return (listOf(phase, String.format(Locale.ROOT, "%.3f", start), String.format(Locale.ROOT, "%.3f", end),
            readings.toString(), sorted.size.toString()) + statistics).joinToString(",")
    }
}
