## Native engine diagnostic

Thread CPU ns/op; medians across three passes and seven samples, grouping measured pointer alignment.

| n | Timing | Alignment | Before | After | Median paired speedup | Pass range |
|---:|---|---|---:|---:|---:|---:|
| 64 | arithmetic | aligned | 21.2 | 20.8 | 1.02x | 1.00–1.03x |
| 64 | arithmetic | misaligned | 24.1 | 23.6 | 1.02x | 0.99–1.03x |
| 64 | reset | aligned | 25.2 | 24.7 | 1.01x | 1.01–1.11x |
| 64 | reset | misaligned | 29.3 | 28.9 | 1.01x | 0.99–1.07x |
| 256 | arithmetic | aligned | 28.0 | 28.1 | 1.00x | 0.98–1.00x |
| 256 | arithmetic | misaligned | 45.0 | 28.9 | 1.56x | 1.55–1.56x |
| 256 | reset | aligned | 39.6 | 39.3 | 0.99x | 0.98–1.03x |
| 256 | reset | misaligned | 57.5 | 40.3 | 1.44x | 1.41–1.45x |
| 4096 | arithmetic | aligned | 215.2 | 219.7 | 0.97x | 0.96–0.98x |
| 4096 | arithmetic | misaligned | 394.2 | 212.4 | 1.87x | 1.80–1.91x |
| 4096 | reset | aligned | 736.5 | 729.3 | 1.02x | 1.00–1.03x |
| 4096 | reset | misaligned | 987.9 | 790.6 | 1.23x | 1.23–1.26x |
| 65536 | arithmetic | aligned | 7384.7 | 7377.3 | 0.99x | 0.97–1.02x |
| 65536 | arithmetic | misaligned | 8204.7 | 7485.9 | 1.09x | 1.08–1.14x |
| 65536 | reset | aligned | 17840.2 | 17368.1 | 1.03x | 0.92–1.09x |
| 65536 | reset | misaligned | 20244.4 | 18355.7 | 1.13x | 0.97–1.14x |

## C leaf and vendor diagnostic

Thread CPU ns/op; final guarded candidate, medians across seven samples and the three misaligned offsets (8, 16, 24 bytes). Each arm rotates within a process.

| n | Timing | Original | Guarded alignment | oneMKL | OpenBLAS |
|---:|---|---:|---:|---:|---:|
| 64 | negate | 6.4 | 6.3 | 13.4 | 7.6 |
| 64 | powers-two | 6.4 | 6.3 | 13.6 | 7.6 |
| 64 | reset | 12.4 | 12.4 | 17.9 | 13.1 |
| 128 | negate | 12.3 | 9.1 | 14.7 | 13.5 |
| 128 | powers-two | 12.3 | 9.1 | 14.9 | 13.6 |
| 128 | reset | 19.2 | 17.5 | 21.5 | 20.3 |
| 256 | negate | 23.5 | 14.4 | 17.9 | 24.8 |
| 256 | powers-two | 23.6 | 14.6 | 18.2 | 25.0 |
| 256 | reset | 34.9 | 26.2 | 30.6 | 35.8 |
| 4096 | negate | 402.8 | 193.3 | 165.9 | 404.2 |
| 4096 | powers-two | 398.5 | 190.7 | 163.7 | 398.2 |
| 4096 | reset | 869.3 | 685.5 | 691.8 | 873.5 |
| 65536 | negate | 8132.4 | 7212.8 | 7204.9 | 8152.8 |
| 65536 | powers-two | 8119.0 | 7195.4 | 7206.2 | 8145.3 |
| 65536 | reset | 18037.9 | 16770.7 | 16790.2 | 17966.0 |

## Standard benchmark suite

Elapsed ns/op; medians across three alternating passes, five samples per pass, five warmups and 200 ms targets. Array addresses are uncontrolled in this suite.

| Case | Native before | Native after | oneMKL | OpenBLAS |
|---|---:|---:|---:|---:|
| scal+256+uniform+timing=arithmetic | 44.6 | 43.8 | 51.0 | 45.1 |
| scal+4096+uniform | 832.4 | 825.4 | 775.4 | 753.0 |
| scal+4096+uniform+timing=arithmetic | 233.6 | 230.0 | 208.1 | 204.1 |
| scal+64+uniform+timing=arithmetic | 40.2 | 39.7 | 45.3 | 39.0 |
| scal+65536+uniform+timing=arithmetic | 7613.1 | 7556.5 | 7604.1 | 8275.1 |
