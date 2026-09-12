## Paired medians in ns/op

| Arithmetic case | Native forced scalar | Native C | oneMKL | OpenBLAS |
|---|---:|---:|---:|---:|
| scal n=4096 | 607.8 | 602.1 | 261.2 | 577.3 |
| spgather n=4096 density=0.25 | 1,246.0 | 333.8 | 295.7 | unsupported |
| spgather n=65536 density=0.25 | 19,820.0 | 7,500.5 | 7,206.5 | unsupported |

- `spgather+4096+sparse-uniform+density=0.25+timing=arithmetic`: median paired speedup 3.71x (passes 2.55–3.82x); native/oneMKL time ratio 1.15 (passes 1.06–2.30).
- `spgather+65536+sparse-uniform+density=0.25+timing=arithmetic`: median paired speedup 2.68x (passes 2.63–3.97x); native/oneMKL time ratio 1.02 (passes 1.01–1.05).

## Initial medians in ns/op

| Case | JVM scalar | JVM C | JVM SIMD | Native | oneMKL | OpenBLAS |
|---|---:|---:|---:|---:|---:|---:|
| scal n=4096 | 925.4 | 865.5 | 843.4 | 844.7 | 1,007.9 | 1,312.1 |
| spgather n=4096 density=0.01 | 593.7 | 734.6 | 765.7 | 641.6 | 1,082.6 | unsupported |
| scal n=64 (arithmetic) | 9.6 | 10.7 | 8.0 | 41.4 | 70.1 | 48.8 |
| scal n=256 (arithmetic) | 18.3 | 20.7 | 18.5 | 47.1 | 87.6 | 56.0 |
| scal n=4096 (arithmetic) | 178.2 | 200.6 | 261.6 | 244.6 | 357.4 | 278.0 |
| scal n=65536 (arithmetic) | 7,463.2 | 8,287.9 | 9,331.3 | 7,987.4 | 14,507.7 | 10,840.8 |
| spgather n=4096 density=0.01 (arithmetic) | 17.3 | 18.9 | 20.1 | 68.9 | 79.9 | unsupported |
| spgather n=4096 density=0.25 (arithmetic) | 222.3 | 242.4 | 238.5 | 1,142.9 | 380.6 | unsupported |
| spgather n=65536 density=0.01 (arithmetic) | 230.2 | 237.5 | 265.7 | 660.6 | 444.3 | unsupported |
| spgather n=65536 density=0.25 (arithmetic) | 5,463.8 | 5,937.5 | 6,926.6 | 13,778.3 | 10,407.2 | unsupported |
