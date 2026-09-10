# oneMKL sparse parity baseline

This is benchmark evidence, not a production dispatch proposal. The bounded contributor profile was measured from
clean commit `b1b0b16728670421291087f5ad303b91ea45e2aa`, based on merged `main` commit
`39e49b390144cb99401f77ab2e252364ba1c7511`. Later commits only retain the evidence and correct benchmark plumbing.

## Runtime and host

- Intel-maintained PyPI package `mkl==2026.1.0`, installed in the isolated virtual environment
  `/home/rasmus/.local/share/koblas-onemkl/venv`.
- Resolved library: `libmkl_rt.so.3`; reported identity: `Intel(R) oneAPI Math Kernel Library Version 2026.1-Product
  Build 20260612 for Intel(R) 64 architecture applications`.
- Linux x86-64, Eclipse Temurin 25.0.1+8-LTS, 12th Gen Intel Core i9-12900H, 20 logical CPUs. Both the
  standard metadata and raw JMH rows identify this measured JVM; the unrelated interactive shell used Java 22.
- Affinity `0,2,4,6`; this pins but does not reserve CPUs. No other koblas report runner was detected, although
  unrelated host activity was not excluded.
- `MKL_NUM_THREADS=1`, `MKL_DYNAMIC=FALSE`, and `OMP_NUM_THREADS=1` for every oneMKL run.

The runtime was installed without changing the system loader configuration:

```bash
python3 -m venv /home/rasmus/.local/share/koblas-onemkl/venv
/home/rasmus/.local/share/koblas-onemkl/venv/bin/pip install mkl==2026.1.0
export LD_LIBRARY_PATH=/home/rasmus/.local/share/koblas-onemkl/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE OMP_NUM_THREADS=1
```

`koblas-bench/report.sh preflight jvm` resolved oneMKL and OpenBLAS 0.3.30 before measurement. The retained
standard bundle was produced by:

```bash
taskset -c 0,2,4,6 koblas-bench/report.sh standard jvm --comparators onemkl --cores 0,2,4,6
```

The runner executed fresh passes in the order built-in, oneMKL, built-in, oneMKL. It validated all 166 rows and
finished in 312.1 seconds. The standard archive has SHA-256
`cf7e9adaec3f0b7f428a65bda09cdb0d21bdf9b5a0c799fa367b4321820781a4`.

The first retained archive incorrectly included the benchmark-only default
`triangleVariant=upper-nontrans-nonunit` in 24 profile-v1 stable IDs. The raw JMH JSON and logs are unchanged. The
current archive explicitly records a derived-metadata migration that removes only that exact default from
`SparseProductHostBenchmark` v1 IDs and rejects non-default variants. Its `case_id_migration` metadata preserves
the superseded archive SHA-256 and every raw-file SHA-256. The resulting 83 built-in IDs exactly match the retained
historical v1 report; developer triangular variants now live in `SparseTriangularVariantBenchmark`.

## Correctness and coverage

Strict JVM tests force the real comparator instead of allowing a missing runtime to skip coverage:

```bash
./gradlew :koblas-bench:check -Pkoblas.oneMklTests=true
./gradlew :koblas:check :koblas-bench:check -Pkoblas.oneMklTests=true -Pkoblas.noSimd=true
```

The explicit scalar implementation is the correctness oracle. Tests cover indexed Level 1, caller-provided
scatter workspace, general and transposed sparse matrix-vector multiplication, lower and upper symmetric
matrix-vector and matrix-matrix multiplication, every transpose pair for sparse-by-dense multiplication, sparse
matrix multiplication, upper sparse and dense SYRK in both orientations, addition with and without transpose, and
lower/upper × transposed/non-transposed × unit/non-unit triangular vector and matrix multiply/solve. Prepared-handle
lifetime is exercised repeatedly. Destination backing buffers are checked where aliasing matters.

Direct oneMKL coverage is available for indexed Level 1, sparse MV/MM, sparse products, upper sparse/dense SYRK,
addition, and left-side triangular operations. Right-side symmetric multiply, triangular multiply, and triangular
solve are benchmark-owned mathematical compositions around left-side calls. Sparse workspace rows are
contract-equivalent compositions; maximum, pivot candidates, checked exceptional diagnostics, and the full
workspace API have no oneMKL counterpart. Native oneMKL bindings are not implemented.

## Standard profile results

The version-1 profile retained 83 matched cases per arm. Its median `koblas/onemkl` time ratio was `1.195`; ratios
above one favor oneMKL. This aggregate mixes dense and sparse rows, so it is only a report-integrity summary.

Representative sparse rows from the two-pass aggregate are:

| operation and workload | oneMKL time | koblas/oneMKL |
| --- | ---: | ---: |
| prepared GEMM, n=257 | 7.15 us | 1.00 |
| prepared GEMV, n=257 | 1.26 us | 0.83 |
| one-shot GEMM, n=257 | 60.60 us | 0.14 |
| one-shot GEMV, n=257 | 42.20 us | 0.03 |
| triangular solve matrix, n=257 | 11.99 us | 0.64 |
| triangular solve vector, n=257 | 3.35 us | 0.56 |
| right triangular multiply composition, n=257 | 50.05 us | 0.08 |
| indexed dot | 41.76 ns | 0.53 |
| indexed gather | 42.05 ns | 0.50 |
| indexed scatter | 46.21 ns | 0.30 |

The clear result is that conversion and handle setup dominate the current one-shot comparator rows. Prepared
measurements are useful external references for optimizing koblas's own sparse algorithms; they do not propose a
production vendor integration or a new prepared API.

## Supplemental selected results

Selected JVM runs used the same runtime, affinity, thread settings, five 500 ms measurement iterations, and one
pass per arm. They were run with `:koblas-bench:jvmSelectedBenchmark` and explicit `bench.include` and `bench.param`
properties. Every successful selected run except the corrected growing-workspace row used source commit
`b1b0b167`; those worktrees were technically dirty only because the completed standard archive was present as an
untracked evidence file. The rejected original growing-workspace run also used `b1b0b167`. Its corrected rerun
used `b1b0b167` plus the uncommitted allocation-classification change later committed unchanged as `825df282`; that
change affects the setup assertion and reporting, not the measured operation. Selected JMH files do not embed Git
state, so this per-run attribution is reconstructed from the retained command chronology and is an explicit
provenance limitation. The selected archive retains JSON, complete logs, `ratios.csv`, and scalar Level-1 results.
Its SHA-256 is
`774a578afd9bfa40e3a09700a324a242624c8c89e13e5e970e6613c28fee4484`.

| selected group | matched rows | median ratio | range |
| --- | ---: | ---: | ---: |
| sparse Level 1, len=256/4096/65536 and density=0.001/0.1 | 30 | 0.89 | 0.13–3.05 |
| workspace equivalent, count=64/512, 50% first touch | 32 | 0.90 | 0.38–1.59 |
| growing workspace, support=512 and scatter=4 | 1 | 0.10 | 0.10 |
| products, n=1024, density=0.001/0.1, banded/skewed | 28 | 0.69 | 0.05–4.52 |
| completion, n=32/257, density=0.01, lower-left | 18 | 0.15 | 0.01–1.82 |
| triangular variants, n=257 | 15 | 0.93 | 0.12–1.88 |
| right triangular solve compositions, n=257 | 3 | 0.34 | 0.15–0.38 |
| C-fallback built-in versus oneMKL, n=257 | 4 | 1.30 | 0.94–1.85 |

At n=1024, prepared GEMM favored oneMKL by 1.25–1.49×, prepared GEMV by 2.28–4.52×, and prepared sparse product
by 1.05–1.35×. Their one-shot equivalents mostly favored koblas because conversion remained timed. Left triangular
solve was near parity or favored oneMKL across the selected storage variants. Right-side compositions remained
2.7–6.9× slower than koblas. The C-fallback comparison resolved `built-in/c-sparse`; oneMKL won three of four
selected prepared/solve rows, while vector solve was approximately tied.

Scalar Level-1 timings are retained as an informative oracle baseline, not as an external-comparator ratio. For
len=65536 and density=0.1, scalar absolute sum was 3.25 us, norm was 3.31 us, and sparse dot-sparse was 37.88 us.

Individual oneMKL workspace primitives measured `0 B/call` after warmup. The growing composition allocates one
sparse-vector wrapper per short scatter and therefore has workload-dependent allocation; an initial run correctly
exposed 70,443 B per composed invocation but was rejected by an over-strict benchmark assertion. The corrected row
measured 48.37 us versus 4.75 us built-in. Both the rejected and corrected logs are retained. Two attempted
right-side symmetric-only runs were rejected by JMH's advisory global lock and are retained as failed, non-evidence
logs; the same operations still have correctness coverage and lower-left timing coverage.

## Ranked koblas optimization follow-ups

1. Optimize koblas's own prepared sparse GEMV, GEMM, and sparse-product algorithms at n=257–1024, using the external
   prepared rows only as benchmark references. Retain the existing prepared API and production engine model.
2. Consolidate koblas sparse support/accumulation leaves so repeated short scatters avoid retained-support overhead.
   Separately, benchmark-only comparator wrappers may cache handles to make external reference rows more honest;
   they must not become a production provider path.
3. Improve koblas's own right-side symmetric and triangular layouts and kernels. The slower oneMKL transpose
   compositions remain benchmark-only diagnostics, not candidates for production dispatch.
4. Add independent interleaved passes around noisy Level-1 and triangular cases. Busy-host evidence remains valid
   when activity, raw samples, and uncertainty are retained; an idle machine is not a gate.
5. Keep Native, ARM, and macOS oneMKL absence as an explicit comparator-coverage gap. Do not create a production
   vendor binding or infer platform policy from JVM results.
6. Preserve scalar-oracle and built-in C coverage for exceptional values, checked workspace diagnostics, aliasing,
   and matrix-property semantics while optimizing the owned kernels.

No production kernel, public API, provider discovery, or dispatch threshold is changed or proposed by this
evidence branch. OpenBLAS and oneMKL remain benchmark-only comparators.
