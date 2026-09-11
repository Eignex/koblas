# PR 01 handoff

Status: verification in progress. This record is updated after baseline capture and independent review.

Implementation session: `01a0910f-6e18-7353-82d3-d1de5ca821a8`, model `gpt-6-astra`, reasoning `high`.
Branch: `codex/sme-baselines`. Production base: `12628d18` (full SHA in source/evidence records).
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

Independent final reviewer must use a fresh context with `gpt-6-astra`, reasoning `xhigh`, read AGENTS.md,
both latest plans, final base/head diff and surrounding sources, and raw logs/CSVs. Final review and final
base/head SHAs will also be saved at `/home/rasmus/Workspaces/koblas-pr01-review.md` so the review can identify
the exact final commit without a self-referential report commit.
