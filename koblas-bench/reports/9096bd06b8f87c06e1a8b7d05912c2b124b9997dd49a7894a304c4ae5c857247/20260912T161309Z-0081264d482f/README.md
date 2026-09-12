# SparseSlices harness validation

All six benchmark arms completed. Background builds and short samples prevent performance conclusions;
overall CPU utilization reached 99.20% (median 58.29%). See [cpu.csv](cpu.csv) and [metadata.txt](metadata.txt).

```bash
taskset -c 4 koblas-bench/capture-report.sh --suite sparse-slices \
  --libraries scalar-slices,onemkl --warmups 5 --samples 3 --target-ms 100 --forks 2
```

Source: `0081264d482f`, unchanged during capture; the dirty marker reflects generated report files.
The [saved cases](cases.txt) follow the [suite contracts](../../../README.md#sparse-slices-with-caller-owned-scratch).
JVM warmups target 100 ms; native/C warmups target 25 ms. All measured samples target 100 ms.

| Arm | Supported / unsupported | Samples |
| --- | ---: | ---: |
| JVM scalar, C, SIMD (each) | 26 / 0 | 156 |
| Native, scalar C slices (each) | 26 / 0 | 78 |
| oneMKL compositions | 19 / 7 | 57 |

Strict joins produced [95 oneMKL pairs](onemkl-comparisons.csv) and
[104 scalar C pairs](scalar-slices-comparisons.csv). Repeat on a quiet host before using their ratios.

Full Gradle checks and docs lint passed, along with JVM/native state tests, JVM allocation checks,
C/oneMKL numerical checks, parser/comparator tests and expected CSV case/sample counts.
