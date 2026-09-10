package com.eignex.koblas.bench;

import java.util.concurrent.TimeUnit;
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
