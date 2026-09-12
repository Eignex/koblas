## Native engine diagnostic

Thread CPU ns/op; medians across three passes and seven samples, grouping measured pointer alignment.

| n | Timing | Alignment | Before | After | Median paired speedup | Pass range |
|---:|---|---|---:|---:|---:|---:|
| 64 | arithmetic | aligned | 19.6 | 19.3 | 1.01x | 1.00–1.02x |
| 64 | arithmetic | misaligned | 22.6 | 22.3 | 1.00x | 0.99–1.04x |
| 64 | reset | aligned | 23.3 | 23.0 | 1.01x | 1.00–1.03x |
| 64 | reset | misaligned | 27.3 | 26.9 | 1.00x | 1.00–1.06x |
| 256 | arithmetic | aligned | 26.2 | 26.1 | 1.00x | 0.99–1.03x |
| 256 | arithmetic | misaligned | 26.9 | 27.2 | 0.98x | 0.98–1.03x |
| 256 | reset | aligned | 36.7 | 36.4 | 1.00x | 1.00–1.03x |
| 256 | reset | misaligned | 38.0 | 38.0 | 0.99x | 0.98–1.02x |
| 4096 | arithmetic | aligned | 205.9 | 179.9 | 1.14x | 1.13–1.16x |
| 4096 | arithmetic | misaligned | 195.1 | 176.2 | 1.11x | 1.10–1.11x |
| 4096 | reset | aligned | 666.2 | 667.7 | 1.00x | 0.93–1.01x |
| 4096 | reset | misaligned | 731.5 | 727.9 | 1.01x | 1.00–1.02x |
| 65536 | arithmetic | aligned | 7058.1 | 7102.2 | 1.00x | 0.97–1.00x |
| 65536 | arithmetic | misaligned | 7156.3 | 7142.1 | 1.00x | 0.98–1.01x |
| 65536 | reset | aligned | 16324.2 | 16560.2 | 0.99x | 0.97–0.99x |
| 65536 | reset | misaligned | 17445.4 | 17548.7 | 1.00x | 0.98–1.00x |

## C prototypes and vendors

Thread CPU ns/op; seven samples. Misaligned rows pool offsets 8, 16 and 24 bytes. Aligned rows use offset zero.

| n | Timing | Alignment | Alignment only | Fixed blocks | OpenBLAS leaf plus alignment | oneMKL | OpenBLAS |
|---:|---|---|---:|---:|---:|---:|---:|
| 64 | negate | aligned | 4.6 | 5.5 | 5.1 | 11.0 | 6.9 |
| 64 | negate | misaligned | 5.9 | 5.9 | 6.6 | 12.6 | 7.1 |
| 64 | powers-two | aligned | 4.4 | 5.5 | 5.1 | 11.2 | 7.3 |
| 64 | powers-two | misaligned | 6.0 | 6.0 | 6.7 | 13.0 | 7.2 |
| 64 | reset | aligned | 8.0 | 9.6 | 8.8 | 15.5 | 10.9 |
| 64 | reset | misaligned | 12.0 | 11.5 | 11.9 | 16.7 | 12.2 |
| 128 | negate | aligned | 7.9 | 6.7 | 7.7 | 13.1 | 8.6 |
| 128 | negate | misaligned | 8.9 | 9.3 | 8.4 | 14.2 | 13.1 |
| 128 | powers-two | aligned | 8.0 | 6.7 | 7.6 | 13.1 | 8.6 |
| 128 | powers-two | misaligned | 9.1 | 9.3 | 8.4 | 14.4 | 13.1 |
| 128 | reset | aligned | 13.6 | 12.4 | 13.4 | 19.1 | 15.0 |
| 128 | reset | misaligned | 16.6 | 16.7 | 16.8 | 20.5 | 18.9 |
| 256 | negate | aligned | 13.8 | 11.2 | 12.7 | 16.0 | 13.3 |
| 256 | negate | misaligned | 14.6 | 13.6 | 12.7 | 17.7 | 24.6 |
| 256 | powers-two | aligned | 14.1 | 11.1 | 12.8 | 15.9 | 14.2 |
| 256 | powers-two | misaligned | 14.8 | 14.1 | 13.6 | 18.1 | 24.8 |
| 256 | reset | aligned | 25.1 | 22.5 | 24.0 | 27.4 | 26.4 |
| 256 | reset | misaligned | 25.7 | 25.1 | 25.4 | 30.0 | 34.7 |
| 4096 | negate | aligned | 191.6 | 163.9 | 158.7 | 159.8 | 160.4 |
| 4096 | negate | misaligned | 186.4 | 159.5 | 156.3 | 160.9 | 393.0 |
| 4096 | powers-two | aligned | 190.0 | 166.5 | 159.5 | 160.1 | 162.3 |
| 4096 | powers-two | misaligned | 185.5 | 161.3 | 157.2 | 160.5 | 392.5 |
| 4096 | reset | aligned | 646.8 | 640.5 | 646.4 | 660.7 | 661.1 |
| 4096 | reset | misaligned | 651.7 | 649.4 | 652.1 | 657.8 | 843.3 |
| 65536 | negate | aligned | 6766.6 | 6816.4 | 6815.0 | 6811.9 | 6799.4 |
| 65536 | negate | misaligned | 6695.9 | 6722.5 | 6725.7 | 6722.9 | 7647.6 |
| 65536 | powers-two | aligned | 6719.7 | 6711.5 | 6719.4 | 6752.8 | 6716.1 |
| 65536 | powers-two | misaligned | 6736.2 | 6751.4 | 6732.0 | 6761.4 | 7636.7 |
| 65536 | reset | aligned | 16236.8 | 16285.4 | 16241.5 | 16364.6 | 16260.9 |
| 65536 | reset | misaligned | 16136.8 | 16032.0 | 16077.3 | 16165.4 | 17317.9 |

## Standard benchmark suite

Elapsed ns/op; medians across three alternating passes, five samples per pass, five warmups and 200 ms targets. Array addresses are uncontrolled in this suite.

| Case | Native original | Native blocks | oneMKL | OpenBLAS |
|---|---:|---:|---:|---:|
| scal+256+uniform+timing=arithmetic | 42.8 | 43.3 | 49.6 | 43.1 |
| scal+4096+uniform | 815.1 | 801.0 | 768.2 | 734.1 |
| scal+4096+uniform+timing=arithmetic | 224.9 | 206.6 | 200.9 | 196.9 |
| scal+64+uniform+timing=arithmetic | 39.1 | 38.7 | 44.1 | 38.4 |
| scal+65536+uniform+timing=arithmetic | 7381.3 | 7367.6 | 7367.4 | 8106.0 |
