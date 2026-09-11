import com.sun.management.OperatingSystemMXBean;
import java.io.BufferedWriter;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Locale;

/** Samples overall CPU utilization independently of the benchmark processes. */
class CpuSampler {
    public static void main(String[] args) throws Exception {
        Path output = Path.of(args[0]);
        Path ready = Path.of(args[1]);
        Path stop = Path.of(args[2]);
        ProcessHandle parent = ProcessHandle.current().parent().orElseThrow();
        var operatingSystem = ManagementFactory.getOperatingSystemMXBean();
        OperatingSystemMXBean cpu = operatingSystem instanceof OperatingSystemMXBean bean ? bean : null;
        try (BufferedWriter writer = Files.newBufferedWriter(output)) {
            writer.write("elapsed_s,cpu_percent\n");
            // The first reading has an undefined observation window.
            if (cpu != null) cpu.getCpuLoad();
            long start = System.nanoTime();
            System.out.println("started_at=" + Instant.now());
            System.out.println("runtime=" + System.getProperty("java.vendor") + "/" + System.getProperty("java.version"));
            int samples = 0;
            while (parent.isAlive() && !Files.exists(stop)) {
                Thread.sleep(1000);
                double load = cpu == null ? -1 : cpu.getCpuLoad();
                String percent = Double.isFinite(load) && load >= 0 && load <= 1
                    ? String.format(Locale.ROOT, "%.2f", load * 100) : "";
                writer.write(String.format(Locale.ROOT, "%.3f,%s%n", (System.nanoTime() - start) / 1e9, percent));
                writer.flush();
                if (++samples == 3) Files.createFile(ready);
            }
        }
    }
}
