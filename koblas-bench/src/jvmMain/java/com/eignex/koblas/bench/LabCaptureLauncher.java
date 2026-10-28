package com.eignex.koblas.bench;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;

/** Runs the capture harness with its reports retained in the lab job's output directory. */
public final class LabCaptureLauncher {
    private LabCaptureLauncher() {}

    /** Passes capture options literally and reserves the output directory supplied by the lab. */
    public static void main(String[] args) throws Exception {
        launch(new String[] {Path.of("koblas-bench/capture-report.sh").toAbsolutePath().toString()}, args);
    }

    static void launch(String[] prefix, String[] args) throws Exception {
        String output = System.getenv("OUT");
        if (output == null || output.isBlank()) {
            throw new IllegalStateException("The lab must supply OUT to retain the benchmark report");
        }
        for (String arg : args) {
            if (arg.equals("--output") || arg.startsWith("--output=")) {
                throw new IllegalArgumentException("The lab supplies the capture output directory");
            }
        }
        var command = new ArrayList<String>();
        command.addAll(Arrays.asList(prefix));
        command.addAll(Arrays.asList(args));
        command.add("--output");
        command.add(Path.of(output, "report").toString());
        Process capture = new ProcessBuilder(command).inheritIO().start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (capture.isAlive()) capture.destroy();
        }, "stop-benchmark-capture"));
        System.exit(capture.waitFor());
    }
}
