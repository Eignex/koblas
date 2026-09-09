# Retained BLAS gap evidence

These selected JVM measurements exercise the new retained dense and sparse rows on an Intel Core i9-12900H.
The host was shared and no reservation or idle period was requested. Each process was pinned to logical CPUs
`2-5`; OpenBLAS was fixed to one thread. The second sparse pass observed another JMH process and therefore used
JMH's documented ignore-lock switch. Its wider `syrk` uncertainty is retained rather than filtered.

Preflight resolved the built-in implementation and OpenBLAS 0.3.30 independently. `libmkl_rt` was unavailable,
so no oneMKL timing or substitute comparator is included. The oneMKL binding still compiles on JVM, and the
macOS ARM64 and Linux ARM64 source sets compile as part of the project gate.

`gemmt-pass-1.json` and `gemmt-pass-2.json` contain reversed-order built-in/OpenBLAS comparisons for a lower
`129x129` result with depth 257. The built-in measurements were `244.595 +/- 10.815 us/op` and
`260.609 +/- 17.351 us/op`; OpenBLAS measured `262.800 +/- 9.710 us/op` and `256.583 +/- 9.624 us/op`.
The confidence intervals overlap, so these runs support parity at this shape rather than a winner claim.

`sparse-pass-1.json` and `sparse-pass-2.json` each contain all nine new sparse benchmark rows at order 129 and
density 0.01. Cross-pass central estimates were stable for addition, direct dense product, prepared and one-shot
symmetric products, and the scaled transposed sparse product. Sparse-result `syrk` showed the expected shared-host
noise: `29.375 +/- 14.641 us/op` and `27.035 +/- 3.257 us/op`. The raw files retain all five samples and JMH
uncertainty for every row.

Before measurement, the full `:koblas:check :koblas-bench:check lintDocs` gate passed, including Linux x64 tests,
Linux ARM64 and macOS ARM64 compilation, ABI checks, documentation lint, and the 97-method benchmark coverage
check. The complete JVM suite also passed with `-Pkoblas.noSimd=true`. Comparator resolution tests execute the
OpenBLAS `gemmt` path against the independently selected built-in engine and instantiate every built-in sparse
completion row.
