package com.eignex.koblas.bench;

import java.nio.file.Path;

/** Measures fixed native buffer alignments beside the owned JVM reduction kernels. */
public final class LabAlignmentReferenceLauncher {
    private LabAlignmentReferenceLauncher() {}

    /** Uses the worker's JDK and keeps the diagnostic under the lab's output directory. */
    public static void main(String[] args) throws Exception {
        LabCaptureLauncher.launch(new String[] {
            "python3", "koblas-bench/experiments/alignment-reference.py", "--java",
            Path.of(System.getProperty("java.home"), "bin", "java").toString()
        }, args);
    }
}
