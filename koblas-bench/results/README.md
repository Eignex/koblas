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
