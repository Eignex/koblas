# Saved benchmark reports

These archives preserve benchmark evidence that would otherwise be deleted with the Gradle build directory.
They may contain machine and checkout details; inspect their metadata before sharing them outside the project.

## Packed syr2k

`syr2k-a7272cee-jvm-20260908.tar.gz` contains the raw JMH JSON used to compare the packed built-in `syr2k`
implementation at commit `a7272ceea6eb11ff39bc07a55ea94cc945a9ac73` with single-threaded OpenBLAS and
oneMKL 2026.1. It includes:

- two independent built-in passes;
- two independent OpenBLAS passes;
- one stable oneMKL pass and two retained noisy oneMKL passes;
- benchmark metadata and the derived timing ratios.

The run used an Intel Core i9-12900H with CPU affinity `0,2,4,6`. The benchmark covered `9x5`, `128x129`,
`257x31`, and `512x127` inputs across both triangle and transpose variants, with three 500 ms warmups and five
500 ms measurements per case.

Verify and inspect the archive with:

```bash
sha256sum koblas-bench/results/syr2k-a7272cee-jvm-20260908.tar.gz
tar -xzf koblas-bench/results/syr2k-a7272cee-jvm-20260908.tar.gz
```

Expected SHA-256:
`b1d3a26a4f5452317cfc2e72d8fff32ecf3aa2616dd05ba17bff6b85da3d4a17`.

## Packed panels

`packed-panels-1dcb7c97-jvm-20260908.tar.gz` preserves two independent JMH passes for the public packed-panel
pack and write helpers at commit `665536e171643356b08de2f75b9c9d76caa3c825`. It also includes an untouched-main
reproduction of the small-depth `gemmTile` allocation bug, two successful post-fix passes from commit
`1dcb7c974ca2f614d4e72fe37e13182682693b6c`, complete console logs, the temporary baseline harness, metadata,
and paired timing results.

The run used an Intel Core i9-12900H with CPU affinity `0,2,4,6` on a busy shared host. It covered depths
`3`, `31`, and `128` at full `8x4` and partial `7x3` tile edges. Every pack and write allocation probe reported
`0 B/call`; confidence intervals and cross-pass variation are retained in the archive. OpenBLAS and oneMKL do
not accept koblas microkernel-packed panels, so there is no meaningful external comparator for these helper-only
operations.

Verify and inspect the archive with:

```bash
sha256sum koblas-bench/results/packed-panels-1dcb7c97-jvm-20260908.tar.gz
tar -xzf koblas-bench/results/packed-panels-1dcb7c97-jvm-20260908.tar.gz
```

Expected SHA-256: `55bf6bebfd18bc77fe27f6476607286fd590cc214b8be67341679dec01a18b39`.
