# koblas-bench

Development measurements for koblas and independent OpenBLAS/oneMKL references. The module is not published.

The authoritative workload is [`cases.txt`](cases.txt). Kotlin benchmarks only koblas and is launched through
Gradle. External references are a standalone Bash and C path: they do not start Gradle, a JVM, or Kotlin. Each
implementation emits raw repeated samples using the same CSV schema. Hardware collection is an independent
shell command and can be run before, after, or without measurements.

## Run

Select an exact koblas engine and either one operation or the complete workload:

```bash
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=all
./gradlew :koblas-bench:jvmSimdBenchmark -Pbench.operation=gemm
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=all
```

The JVM benchmark toolchain is intentionally JDK 25. Its actual vendor and version are recorded in every CSV
row; this is the benchmark process runtime and may differ from the JDK that launched Gradle. `jvm-c`,
`jvm-simd`, and `native` resolve an exact immutable provider through `BuiltinKernels`. A missing requested engine
fails rather than selecting another one. Normal koblas algorithm fallbacks inside that engine remain part of the
measurement. Native resolves to Linux x86-64 or macOS arm64 on the current host and fails elsewhere.

Useful bounded timing controls are `-Pbench.warmups=3`, `-Pbench.samples=5`, `-Pbench.targetMs=100`,
`-Pbench.pass=1`, and `-Pbench.output=path.csv`. Calibration happens before measured samples. Destructive
operations restore their fixture on every invocation. Prepared sparse handles are created before timing and
closed afterward; `mode=oneshot` includes the operation's ordinary preparation/conversion lifetime. The result
is consumed so the work stays observable. Benchmarks do not inspect or classify allocation: allocation
regressions belong in focused unit tests, while allocations intrinsic to one-shot work naturally remain timed.

Run the fast, representative 11-case smoke suite first:

```bash
koblas-bench/reference-smoke.sh --libraries openblas
koblas-bench/reference-smoke.sh --libraries openblas,onemkl --output /tmp/koblas-reference-smoke
```

The smoke command uses [`smoke-cases.txt`](smoke-cases.txt), whose case IDs are checked as an exact subset of
the authoritative workload. It forces zero warmups, one sample and a 1 ms calibration target. Use it to verify
the requested libraries, fixture goldens, numerical preflight and CSV path before committing to the full run.

Then run one or both external libraries through the complete workload into a fresh directory:

```bash
koblas-bench/reference.sh --libraries openblas,onemkl
koblas-bench/reference.sh --libraries openblas --output /tmp/koblas-reference
```

Both commands print their output directory. They create `openblas.csv` and/or `onemkl.csv`; measurements from two
libraries are never combined. Vendor threads are fixed to one. OpenBLAS must be linkable as `-lopenblas`.
oneMKL defaults to `/home/rasmus/.local/share/koblas-onemkl/venv/lib/libmkl_rt.so.3`; set
`ONEMKL_LIBRARY=/path/to/libmkl_rt.so.3` to use another runtime. An explicitly requested missing library is an
error. The C executable verifies fixture goldens and representative dense (and, for oneMKL, sparse) numerical
agreement before measuring.

Collect hardware independently when useful:

```bash
koblas-bench/tools/hardware.sh > hardware.txt
```

Compare matching rows without probing hardware or libraries:

```bash
python3 koblas-bench/tools/compare.py openblas.csv jvm-c.csv onemkl.csv
```

The comparator joins only identical case IDs, workload/fixture versions, timing modes, and thread counts. It
does not invent timings or ratios for unsupported rows. Busy-machine samples are valid noisy evidence; no runner
reserves cores, waits for an idle host, or changes another process.

## Case language

Each non-comment line has one canonical concrete case:

```text
# operation+dimensions+fixture[+option=value...]
dot+4096+uniform
gemm+32x21x48+uniform
gemm+129x31x257+uniform+transA=T
trsm+64x16+triangular+side=L+uplo=L+transA=N+diag=N
spgemv+257x129+sparse-uniform+density=0.01+mode=prepared
```

Blank lines and whole-line comments are allowed. There are no ranges, macros, includes, or expressions. Options
use canonical order `density`, `mode`, `physical`, `side`, `uplo`, `transA`, `transB`, `diag`. Both parsers reject
unknown operations/fixtures/options, non-positive or wrong-count dimensions, invalid option values, duplicate or
out-of-order options, incompatible fixtures/modes, redundant optional defaults, and duplicate complete cases.
Structural flags are mandatory for triangular/symmetric/rank cases, as are sparse density/mode and packed physical
shape, so their default-valued fields remain meaningful canonical fields rather than aliases. The complete line is
the CSV case ID. Operation selection filters this list and never expands a separate workload.

Dimensions have mathematical meaning:

| family | dimensions | contract |
| --- | --- | --- |
| vector and workspace | `length` | logical vector or workspace dimension |
| GEMV / GER | `m x n` | matrix/result shape independent of transpose storage |
| GEMM | `m x n x k` | `op(A)` is `m x k`, `op(B)` is `k x n` |
| symmetric rank / GEMMT | `n x k` | symmetric result order and update depth |
| TRSM / TRMM | `m x n` | shape of B; `side` determines the triangle order |
| packed GEMM/fusion | `rows x columns x depth` | logical edge; `physical` records padded tile shape |
| packed TRSM | `rows x order` | logical edge inside the recorded physical tile |
| left/right layout | `rows x depth` / `depth x columns` | logical panel copied to or from the physical layout |
| sparse GEMV | `m x n` | CSC matrix shape |
| sparse-dense / sparse-sparse product | `m x n x k` | result shape and inner dimension |

Defaults are alpha `0.875`, beta `-0.25`, `side=L`, `uplo=L`, `transA=N`, `transB=N`, and `diag=N`.
Destinations start from deterministic `uniform` values and are reset for every destructive invocation. Layout
and packed cases state their physical tile explicitly. A target whose packed tile differs emits `unsupported`
for that case rather than pretending different physical work has the same ID.

## Fixtures

Fixture version 1 uses SplitMix64 with seed `0x243f6a8885a308d3`. Every `Long`/`uint64_t` operation wraps modulo
2^64. Stream `q` begins at `seed XOR (q * 0x9e3779b97f4a7c15)`. Each generated value is
`(nextUInt64 >> 11) * 2^-52 - 1`, exactly representable apart from the final subtraction. Dense matrices are
column-major. Operands use independent fixed stream numbers.

`triangular` clears the unstored triangle and sets diagonal `i` to `2 + abs(value(i,i))`; unit-diagonal calls
still do not read it. `sparse-uniform` rounds per-column support as `floor(rows*density + 0.5)`, clamped to
`1..rows`. It ranks rows by the signed 64-bit SplitMix finalizer of the column seed plus
`row * 0x9e3779b97f4a7c15`, selects the lowest ranks, and stores selected rows ascending in validated CSC.
`sparse-triangular` removes positions outside the selected triangle and inserts a safe diagonal when absent.

Both languages independently assert these goldens before their relevant checks:

| fixture | golden |
| --- | --- |
| dense stream 1, length 16, FNV-1a over little-endian Double bits | `173cc3a80546954d` |
| sparse 17x5, density 0.2, stream 2, value digest | `e6ac2de9cae9ebe8` |
| same sparse fixture column pointers | `0,3,6,9,12,15` |

## CSV and timing

Schema 2 columns are `schema`, `case`, `implementation`, `workload_version`, `fixture_version`, `pass`,
`sample`, `operations`, `elapsed_ns`, `ns_per_op`, `unit`, `status`, `comparison_kind`, `timing_mode`,
`source_commit`, `dirty`, `runtime`, and `threads`. Numbers are locale-independent and CSV fields are escaped.
Every successful sample is retained. Unsupported cases carry no elapsed value. Supported-call failures abort the
run after recording failure on Kotlin; they are never converted into another implementation.

External calls measure vendor execution without Kotlin binding overhead. Conversion from koblas packed layout to
ordinary column-major matrices is staged before arithmetic-only vendor timing. Vendor-private packed formats are
never assumed compatible. See [`coverage.md`](coverage.md) for the complete direct/composed/partial/unsupported
audit and timing boundaries.

## Verify

```bash
koblas-bench/reference/test.sh
./gradlew :koblas-bench:jvmTest
./gradlew :koblas-bench:jvmCBenchmark -Pbench.operation=gemm -Pbench.samples=1 -Pbench.targetMs=1
./gradlew :koblas-bench:nativeBenchmark -Pbench.operation=dot -Pbench.samples=1 -Pbench.targetMs=1
```

[`example.csv`](example.csv) is a compact schema example from this implementation worktree. Its `dirty=true`
provenance is intentional, and its one-sample smoke timings demonstrate format and engine identity only; they are
not performance evidence.
