package com.eignex.koblas.bench;

import java.util.concurrent.TimeUnit;
import java.lang.management.ManagementFactory;
import com.sun.management.HotSpotDiagnosticMXBean;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
public class KoblasJmhBenchmark {
    @Param({"unset"})
    public String benchmarkMode;

    @Param({"unset"})
    public String caseId;

    private JvmCaseWork work;

    @Setup(Level.Trial)
    public void setup() {
        work = JvmBenchmarkBridge.create(benchmarkMode, caseId, System.getProperty("koblas.bench.cases"));
        String collectors=String.join("|",ManagementFactory.getGarbageCollectorMXBeans().stream().map(c->c.getName()).toList());
        System.out.println("fork_runtime case="+caseId+" mode="+benchmarkMode+" java="+Runtime.version()
            +" collectors="+collectors+" max_heap_bytes="+Runtime.getRuntime().maxMemory()
            +" region_bytes="+runtimeOption("G1HeapRegionSize")
            +" object_alignment="+runtimeOption("ObjectAlignmentInBytes")
            +" compact_headers="+runtimeOption("UseCompactObjectHeaders"));
    }

    private static String runtimeOption(String name) {
        try {
            return ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class).getVMOption(name).getValue();
        } catch (RuntimeException unavailable) {
            return "unavailable";
        }
    }

    @Benchmark
    public void run(Blackhole blackhole) {
        blackhole.consume(work.run());
    }

    @TearDown(Level.Trial)
    public void close() {
        work.close();
    }
}
