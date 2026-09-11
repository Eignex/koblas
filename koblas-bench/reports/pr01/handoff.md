# PR 01 handoff

Status: local G1/G3/G5 evidence complete; final independent verdict is recorded at the external review path below.
M4/Arm measurement evidence remains unavailable and is not claimed complete.

Implementation session: `01a0910f-6e18-7353-82d3-d1de5ca821a8`, model `gpt-6-astra`, reasoning `high`.
Branch: `codex/sme-baselines`. Production base: `12628d1859b25dcfbbd64eb49fc5e905f87bb4d8`.
Scope: PR 01 only. No production dispatch changes, PR 02 work, oneMKL parity project, push, or merge.
Backward API compatibility is not a requirement. The current internal JVM FFM binding is accessed via a
benchmark-only friend module; PR 07 removes this adapter when the exact block interface replaces it.

Requirements read from the latest user-owned files, not the older planning branch:

- `/home/rasmus/Workspaces/koblas-sme.md`: SHA256 `c7b91389c9ff88c277ef579634978bcd699cc47492db6e1f0f593f9ddb199c58`
- `/home/rasmus/Workspaces/koblas-sme-steps.md`: SHA256 `dfa3ca7bde3584836a48427462c273062c988241c478908095005cad90595529`

Contract and harness changes: see `../../contracts.md` and `../../packed-cases.md`. Workload 5, fixture 2,
CSV schema 4 deliberately start fresh evidence; historical reports are untouched. The shared workload has
143 cases, including logical blocks, skinny/depth-tail GEMM and single/many-RHS solves. The comparator keeps
fixed configurations distinct and allows complete logical block strategies to differ without pooling samples.

G1/G3: repository checks passed on Linux x86-64; logs are adjacent. `fixedConfigurationTest` also passed with
packing defaults 3/5/7, C tile crossovers 1000000 and `-XX:MaxVectorSize=16`. Both C and narrower SIMD work are
checked against every scalar-reference output element. View overlap rejection is checked before mutation;
beta-zero strided output guards are checked bitwise. An initial test incorrectly assumed borrowed-view alias
staging; source/KDoc confirmed rejection, and the test now pins that contract. No production change was made.

Hardware: local Intel Core i9-12900H, Linux x86-64. Only the `github` SSH alias is configured; no M4 Mac mini
connection was supplied or found. macOS/Arm execution, SME/SME2 execution, and Apple performance evidence are
missing. Cross-target compilation/skips do not count as execution. These missing target measurements remain
outstanding; no Arm support, speedup, threshold, or oneMKL parity claim follows from this PR.

Independent final reviewer uses a fresh context with `gpt-6-astra`, reasoning `xhigh`, read AGENTS.md,
both latest plans, final base/head diff and surrounding sources, and raw logs/CSVs. Final review and final
base/head SHAs will also be saved at `/home/rasmus/Workspaces/koblas-pr01-review.md` so the review can identify
the exact final commit without a self-referential report commit.


## Captured baseline evidence

Hardware fingerprint: `f342c3192a41a07cf0454d7d4d5324f8289cc01329bcc9c990b986dcdd2f5922`.
Paths below are under the adjacent fingerprint directory in `koblas-bench/reports/`:

- `20260911T153950Z-a038f25ac668`: full 143-case capture, source
  `a038f25ac66821574b30392dc1d3b189cc287e2c`. Each Koblas mode has 585 samples across 117 supported
  cases and 26 unsupported cases. OpenBLAS has 280 samples across 56 supported cases and 87 unsupported.
- `20260911T155812Z-50f1356ff734`: refreshed 52-case packed suite after pinning every helper recipe argument,
  source `50f1356ff734c61f359e46e821b0f580d6549e73`. Each Koblas mode has 130 samples across 26 supported
  cases and 26 unsupported shapes. OpenBLAS has 80 samples across 16 supported cases and 36 unsupported.

Both runs completed sequentially on the actual local host: JVM scalar, JVM C, JVM SIMD, Native C, then
OpenBLAS. Settings: 3 warmups, 5 samples, target 200 ms, one JVM fork, pass 1. Every raw sample and JMH log,
case snapshot, runtime/compiler provenance and status file is preserved. JVM runtime is Eclipse Adoptium
25.0.1; Kotlin is 2.4.10; Native uses Clang 21.1.6. Both source.patch files are empty. CSV dirty=true reflects
report creation; the only later source change during the refreshed capture was test-only geometry selection.
No benchmark implementation changed after its recorded source commit. Reports are not relabelled.

The final source also contains the reviewer-requested geometry-selector test, which cannot change measured
work. Full packed data was recaptured because explicit helper arguments can affect timed call overhead.
The earlier full capture remains useful for unchanged full-operation policy baselines and retains its own SHA.

`audit-report.py` and `report-audit.txt` check case membership, schema/workload/fixture versions, source SHAs,
sample counts, positive finite metrics and elapsed/operation arithmetic. Adjacent comparison CSVs retain both
physical strategies and actual kernel IDs; `comparison-commands.txt` reproduces fixed scalar/C/Native and
logical C/SIMD/vendor comparisons. These are initial baselines, not threshold calibrations: some within-case
spreads approach 2.6x, the CPU is not pinned and host exclusivity/thermal stability were not enforced. No
speedup, default-policy change, instruction-throughput inference or oneMKL performance-parity claim is made.
The oneMKL smoke CSV is library/runner validation only, not performance evidence.

## Final checks and review resolutions

- `check.log`: `./gradlew :koblas:check :koblas-bench:check lintDocs` passed after all source/test fixes.
- `check-c.log`: the same command with `-Pkoblas.noSimd=true` passed.
- `vendor-test.log`: OpenBLAS smoke, parser rejection (every required field), full shared workload and comparator tests passed.
- `comparator-tests.log`: all seven comparator tests passed, including the strict timing selector and exclusion of policy rows in fixed mode.
- `onemkl-smoke.log` / `onemkl-smoke.csv`: existing oneMKL runner compiled and executed its eleven smoke cases.
- Retained test XML shows the final format, geometry-selector and strided-reference tests; final G1 includes
  `fixedConfigurationTest` with altered tuning defaults and vector width. New JVM tests remain under 300 ms.
- The documented strict prepacked vendor comparison passed directly on the raw full reports. Fixed mode
  also passed for the refreshed scalar/C and C/Native reports; logical mode passed for C/SIMD block timings.

Fresh reviewer: `/root/independent_review`, spawned with `fork_turns=none`, model `gpt-6-astra`, reasoning
`xhigh`. The tool returned no opaque reviewer session ID; none is invented. Three P2 findings were resolved:

1. Pin all fixed semantic helper arguments and independently check the immutable packed formulas bitwise.
2. Make the README strict comparison select the complete prepacked boundary, leaving raw vendor rows honestly incompatible.
3. Select only explicitly listed 4x4/8x4 test geometries; do not assume every other species is 8 rows.

The reviewer accepted all three source fixes by inspection and is assessing the final base/head pair with the
completed raw evidence. The definitive final verdict and exact head SHA are saved at
`/home/rasmus/Workspaces/koblas-pr01-review.md`; keeping this final attestation outside its reviewed commit
avoids a self-referential commit hash. No push, merge, or next implementation step has been performed.
