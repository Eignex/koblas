package com.eignex.koblas.bench;

/** Captures a pinned JVM baseline, this revision and native oneMKL in rotating rounds on one worker. */
public final class LabComparisonLauncher {
    private LabComparisonLauncher() {}

    /** Retains the comparison and each completed round in the lab output directory. */
    public static void main(String[] args) throws Exception {
        LabCaptureLauncher.launch(new String[] {"python3", "koblas-bench/compare-revisions.py"}, args);
    }
}
